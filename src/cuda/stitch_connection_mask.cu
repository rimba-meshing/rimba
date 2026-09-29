#include "stitch_connection_mask.hpp"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <cmath>

namespace ri_meshing::cuda_detail {
namespace {

struct ConnectionMaskBuffers {
    const unsigned char* new_valid{};
    const int* effective_new_vim{};
    const unsigned char* overlap_allowed{};
    const unsigned char* final_allowed{};
    const unsigned char* thin_removed{};
    AccumulatedDepthView old_depth;
    unsigned char* output{};
    unsigned char* old_support_output{};
    unsigned char* new_support_output{};
};

__device__ bool effective_new_pixel_present(
    const ConnectionMaskBuffers& buffers,
    int pixel
)
{
    if (buffers.effective_new_vim != nullptr) {
        // A propagated label does not undo an overlap/thin-filter rejection
        // for a pixel that was an actual current-scan candidate. Empty RI
        // pixels may still use the filled VIM as support.
        if (buffers.new_valid[pixel] != 0 && buffers.final_allowed[pixel] == 0) {
            return false;
        }
        return buffers.effective_new_vim[pixel] >= 0;
    }
    return buffers.new_valid[pixel] != 0 && buffers.final_allowed[pixel] != 0;
}

struct ConnectionMaskDimensions {
    int height{};
    int width{};
};

__device__ float old_depth_at(
    AccumulatedDepthView depth,
    int row,
    int col,
    ConnectionMaskDimensions dimensions
)
{
    if (depth.values == nullptr) {
        return __int_as_float(0x7f800000);
    }
    const int source_row = depth.bottom_up_normalized ? dimensions.height - 1 - row : row;
    const float value = depth.values[source_row * dimensions.width + col];
    if (!depth.bottom_up_normalized) {
        return value;
    }
    return value >= 0.999999F ? __int_as_float(0x7f800000) : value * 4096.0F;
}

__device__ int wrap_column(int col, int width)
{
    return col < 0 ? col + width : (col >= width ? col - width : col);
}

__device__ bool old_pixel_present(
    AccumulatedDepthView depth,
    int row,
    int col,
    ConnectionMaskDimensions dimensions
)
{
    return row >= 0 && row < dimensions.height
        && isfinite(old_depth_at(depth, row, col, dimensions));
}

__device__ bool has_old_eight_neighbour(
    const ConnectionMaskBuffers& buffers,
    int row,
    int col,
    ConnectionMaskDimensions dimensions
)
{
    for (int dr = -1; dr <= 1; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= dimensions.height) {
            continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) {
                continue;
            }
            const int nc = wrap_column(col + dc, dimensions.width);
            if (old_pixel_present(buffers.old_depth, nr, nc, dimensions)) {
                return true;
            }
        }
    }
    return false;
}

__device__ bool has_accepted_eight_neighbour(
    const ConnectionMaskBuffers& buffers,
    int row,
    int col,
    ConnectionMaskDimensions dimensions
)
{
    for (int dr = -1; dr <= 1; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= dimensions.height) {
            continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) {
                continue;
            }
            const int nc = wrap_column(col + dc, dimensions.width);
            const int neighbour = nr * dimensions.width + nc;
            if (effective_new_pixel_present(buffers, neighbour)) {
                return true;
            }
        }
    }
    return false;
}

__device__ bool is_base_connection_pixel(
    const ConnectionMaskBuffers& buffers,
    int row,
    int col,
    ConnectionMaskDimensions dimensions
)
{
    const int pixel = row * dimensions.width + col;
    const bool raw_new = buffers.new_valid[pixel] != 0;
    const bool effective_new = effective_new_pixel_present(buffers, pixel);
    if ((!raw_new && !effective_new)
        || old_pixel_present(buffers.old_depth, row, col, dimensions)) {
        return false;
    }
    const bool stitch_rejected = raw_new && (
        buffers.overlap_allowed[pixel] == 0 || buffers.thin_removed[pixel] != 0
    );
    const bool old_neighbour = has_old_eight_neighbour(
        buffers, row, col, dimensions
    );
    const bool accepted_neighbour = has_accepted_eight_neighbour(
        buffers, row, col, dimensions
    );
    return (stitch_rejected && old_neighbour && accepted_neighbour)
        || (effective_new && old_neighbour);
}

__global__ void build_connection_mask_kernel(
    ConnectionMaskBuffers buffers,
    ConnectionMaskDimensions dimensions
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = dimensions.height * dimensions.width;
    if (pixel >= pixel_count) {
        return;
    }
    const int row = pixel / dimensions.width;
    const int col = pixel - row * dimensions.width;
    const bool old_here = old_pixel_present(buffers.old_depth, row, col, dimensions);
    buffers.old_support_output[pixel] = old_here ? 1 : 0;
    // The open-side ray may walk through accepted new pixels only. Rejected
    // candidates are not part of the post-overlap new mesh and must not block
    // a ray coming from the old side.
    const bool new_support = effective_new_pixel_present(buffers, pixel) && !old_here;
    buffers.new_support_output[pixel] = new_support ? 1 : 0;
    buffers.output[pixel] = (buffers.new_valid[pixel] != 0
        || effective_new_pixel_present(buffers, pixel))
        && is_base_connection_pixel(buffers, row, col, dimensions) ? 1 : 0;
}

__device__ bool has_marked_eight_neighbour(
    const unsigned char* source,
    int row,
    int col,
    ConnectionMaskDimensions dimensions
)
{
    for (int dr = -1; dr <= 1; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= dimensions.height) {
            continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) {
                continue;
            }
            const int nc = wrap_column(col + dc, dimensions.width);
            if (source[nr * dimensions.width + nc] != 0) {
                return true;
            }
        }
    }
    return false;
}

__global__ void expand_connection_mask_kernel(
    ConnectionMaskBuffers buffers,
    ConnectionMaskDimensions dimensions,
    const unsigned char* source,
    unsigned char* target
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = dimensions.height * dimensions.width;
    if (pixel >= pixel_count) {
        return;
    }
    if (source[pixel] != 0) {
        target[pixel] = 1;
        return;
    }
    const int row = pixel / dimensions.width;
    const int col = pixel - row * dimensions.width;
    const bool old_here = old_pixel_present(buffers.old_depth, row, col, dimensions);
    const bool stitch_rejected = buffers.overlap_allowed[pixel] == 0
        || buffers.thin_removed[pixel] != 0;
    const bool expandable = buffers.new_valid[pixel] != 0
        && !old_here && stitch_rejected;
    target[pixel] = expandable
        && has_marked_eight_neighbour(source, row, col, dimensions) ? 1 : 0;
}

}  // namespace

void build_stitch_connection_mask(StitchConnectionMaskRequest request)
{
    const int pixel_count = request.height * request.width;
    resize_with_headroom(request.execution.workspace.stitch_connection_mask, static_cast<std::size_t>(pixel_count));
    resize_with_headroom(request.execution.workspace.stitch_connection_old_support, static_cast<std::size_t>(pixel_count));
    resize_with_headroom(request.execution.workspace.stitch_connection_new_support, static_cast<std::size_t>(pixel_count));
    resize_with_headroom(request.execution.workspace.stitch_connection_next_frontier, static_cast<std::size_t>(pixel_count));
    constexpr int block = 256;
    const int grid = (pixel_count + block - 1) / block;
    const ConnectionMaskDimensions dimensions{request.height, request.width};
    const ConnectionMaskBuffers buffers{
        request.new_valid,
        request.effective_new_vim,
        request.overlap_allowed,
        request.final_allowed,
        request.thin_removed,
        request.accumulated_depth,
        thrust::raw_pointer_cast(request.execution.workspace.stitch_connection_mask.data()),
        thrust::raw_pointer_cast(
            request.execution.workspace.stitch_connection_old_support.data()
        ),
        thrust::raw_pointer_cast(
            request.execution.workspace.stitch_connection_new_support.data()
        ),
    };
    timed_cuda_stage_void(request.execution.timing, "stitch.connection_mask", [&] {
        build_connection_mask_kernel<<<grid, block, 0, request.execution.stream>>>(
            buffers,
            dimensions
        );
        CUDA_CHECK(cudaGetLastError());
        for (int hop = 0; hop < request.frontier_expand_radius; ++hop) {
            expand_connection_mask_kernel<<<grid, block, 0, request.execution.stream>>>(
                buffers,
                dimensions,
                thrust::raw_pointer_cast(
                    request.execution.workspace.stitch_connection_mask.data()
                ),
                thrust::raw_pointer_cast(
                    request.execution.workspace.stitch_connection_next_frontier.data()
                )
            );
            CUDA_CHECK(cudaGetLastError());
            request.execution.workspace.stitch_connection_mask.swap(
                request.execution.workspace.stitch_connection_next_frontier
            );
        }
    });
}

void materialize_stitch_connection_mask(
    StitchConnectionMaskMaterializeRequest request
)
{
    const int pixel_count = request.height * request.width;
    CUDA_CHECK(cudaStreamSynchronize(request.execution.stream));
    request.host_output.height = request.height;
    request.host_output.width = request.width;
    request.host_output.pixels.resize(static_cast<std::size_t>(pixel_count));
    request.host_output.old_support.resize(static_cast<std::size_t>(pixel_count));
    request.host_output.new_support.resize(static_cast<std::size_t>(pixel_count));
    CUDA_CHECK(cudaMemcpy(
        request.host_output.pixels.data(),
        thrust::raw_pointer_cast(request.execution.workspace.stitch_connection_mask.data()),
        static_cast<std::size_t>(pixel_count) * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        request.host_output.old_support.data(),
        thrust::raw_pointer_cast(
            request.execution.workspace.stitch_connection_old_support.data()
        ),
        static_cast<std::size_t>(pixel_count) * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        request.host_output.new_support.data(),
        thrust::raw_pointer_cast(
            request.execution.workspace.stitch_connection_new_support.data()
        ),
        static_cast<std::size_t>(pixel_count) * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));
}

}  // namespace ri_meshing::cuda_detail
