#include "scan_internal.cuh"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"
#include "ri_thin_filter.hpp"

#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cmath>
#include <type_traits>
#include <utility>

namespace ri_meshing::cuda_detail {
namespace {

constexpr float kPiF = 3.1415926535897932384626433832795F;

struct OverlapBuffers {
    const unsigned char* valid{};
    const Vec3f* pixel_xyz{};
    AccumulatedDepthView accumulated_depth;
    unsigned char* allowed{};
};

template <typename Scalar>
struct OverlapParams {
    RiProjection<Scalar> projection;
    Scalar neighbor_tolerance_sq{};
};

struct VertexBuildBuffers {
    const unsigned char* valid{};
    const int* offsets{};
    const Vec3f* pixel_xyz{};
    Vec3f* vertices{};
    int* vertex_rows{};
    int* vertex_cols{};
    float* vertex_ranges{};
    int* vim{};
};

struct VertexBuildLayout {
    RiDimensions dimensions;
    int pixel_count{};
    int old_vertex_count{};
};

static_assert(std::is_trivially_copyable_v<OverlapBuffers>);
static_assert(std::is_trivially_copyable_v<OverlapParams<float>>);
static_assert(std::is_trivially_copyable_v<VertexBuildBuffers>);
static_assert(std::is_trivially_copyable_v<VertexBuildLayout>);

__device__ float device_range(Vec3f point)
{
    return sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
}

__device__ bool depth_pixel_valid(float range)
{
    return isfinite(range);
}

__device__ float accumulated_depth_at(
    AccumulatedDepthView depth,
    int row,
    int col,
    RiDimensions dimensions
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

__device__ Vec3f ri_pixel_to_xyz(
    int row,
    int col,
    float point_range,
    RiProjection<float> projection
)
{
    const float phi = projection.fov_min
        + (static_cast<float>(row) / static_cast<float>(projection.dimensions.height)) * projection.fov_range;
    const float theta = (static_cast<float>(col) / static_cast<float>(projection.dimensions.width))
        * 2.0F * kPiF - kPiF;
    const float xy = point_range * cosf(phi);
    return Vec3f{xy * cosf(theta), xy * sinf(theta), point_range * sinf(phi)};
}

__global__ void mark_new_vertex_allowed_kernel(
    OverlapBuffers buffers,
    OverlapParams<float> params
)
{
    const int height = params.projection.dimensions.height;
    const int width = params.projection.dimensions.width;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = height * width;
    if (pixel >= pixel_count) {
        return;
    }
    if (buffers.valid[pixel] == 0) {
        buffers.allowed[pixel] = 0;
        return;
    }

    const int row = pixel / width;
    const int col = pixel - row * width;
    if (depth_pixel_valid(accumulated_depth_at(
        buffers.accumulated_depth, row, col, params.projection.dimensions
    ))) {
        buffers.allowed[pixel] = 0;
        return;
    }

    const Vec3f point = buffers.pixel_xyz[pixel];
    for (int dr = -1; dr <= 1; ++dr) {
        const int nr = row + dr;
        if (nr < 0 || nr >= height) {
            continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) {
                continue;
            }
            int nc = col + dc;
            if (nc < 0) {
                nc += width;
            } else if (nc >= width) {
                nc -= width;
            }
            const float old_range = accumulated_depth_at(
                buffers.accumulated_depth, nr, nc, params.projection.dimensions
            );
            if (!depth_pixel_valid(old_range)) {
                continue;
            }
            const Vec3f old_point = ri_pixel_to_xyz(nr, nc, old_range, params.projection);
            const float dx = point.x - old_point.x;
            const float dy = point.y - old_point.y;
            const float dz = point.z - old_point.z;
            if (dx * dx + dy * dy + dz * dz < params.neighbor_tolerance_sq) {
                buffers.allowed[pixel] = 0;
                return;
            }
        }
    }
    buffers.allowed[pixel] = 1;
}

__global__ void make_flags_kernel(const unsigned char* valid, int* flags, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        flags[idx] = valid[idx] != 0;
    }
}

__global__ void build_vertices_kernel(
    VertexBuildBuffers buffers,
    VertexBuildLayout layout
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= layout.pixel_count) {
        return;
    }
    if (buffers.valid[pixel] == 0) {
        buffers.vim[pixel] = -1;
        return;
    }
    const int vertex_id = layout.old_vertex_count + buffers.offsets[pixel];
    const int row = pixel / layout.dimensions.width;
    const int col = pixel - row * layout.dimensions.width;
    buffers.vim[pixel] = vertex_id;
    buffers.vertices[vertex_id] = buffers.pixel_xyz[pixel];
    buffers.vertex_rows[vertex_id] = row;
    buffers.vertex_cols[vertex_id] = col;
    buffers.vertex_ranges[vertex_id] = device_range(buffers.pixel_xyz[pixel]);
}

class VertexCompletionPass {
public:
    VertexCompletionPass(
        CudaPixelScanPrepass prepass,
        const SparseMeshingConfig& config,
        CudaExecutionContext execution,
        VertexCompletionOptions completion
    )
        : prepass_(prepass), config_(config), execution_(std::move(execution)),
          completion_(completion), options_(completion.scan),
          fov_min_(static_cast<float>(config.vertical_fov.min_rad)),
          fov_range_(static_cast<float>(
              config.vertical_fov.max_rad - config.vertical_fov.min_rad
          ))
    {
        select_input_buffers();
    }

    CudaRiFrame run()
    {
        reject_overlap();
        prepare_final_allowed_mask();
        emit_timeline("vertex_build_start");
        const int vertex_count = compute_vertex_offsets();
        CudaRiFrame frame = allocate_frame(vertex_count);
        build_vertices(frame);
        emit_timeline("vertex_build_done");
        return frame;
    }

private:
    static constexpr int kBlock = 256;

    int grid_for_pixels() const
    {
        return (prepass_.pixel_count + kBlock - 1) / kBlock;
    }

    void emit_timeline(const char* event)
    {
        if (completion_.emit_timeline && execution_.timeline) {
            execution_.timeline(event);
        }
    }

    void select_input_buffers()
    {
        CudaWorkspace& workspace = execution_.workspace;
        valid_for_vertices_ = thrust::raw_pointer_cast(workspace.valid.data());
        pixel_xyz_ = thrust::raw_pointer_cast(workspace.filled_pixel_xyz.data());
    }

    OverlapBuffers overlap_buffers()
    {
        CudaWorkspace& workspace = execution_.workspace;
        return OverlapBuffers{
            valid_for_vertices_, pixel_xyz_, *options_.accumulated_depth,
            thrust::raw_pointer_cast(workspace.new_vertex_allowed.data()),
        };
    }

    void launch_overlap_rejection()
    {
        const OverlapBuffers buffers = overlap_buffers();
        const MeshScalar tolerance = config_.overlap_neighbor_euclidean_tol;
        mark_new_vertex_allowed_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
            buffers,
            OverlapParams<float>{
                {{prepass_.height, prepass_.width}, fov_min_, fov_range_},
                static_cast<float>(tolerance * tolerance),
            }
        );
    }

    void reject_overlap()
    {
        if (options_.accumulated_depth == nullptr) {
            return;
        }
        emit_timeline("mark_allowed_start");
        timed_cuda_stage_void(execution_.timing, "scan.mark_new_vertex_allowed", [&] {
            CudaWorkspace& workspace = execution_.workspace;
            resize_with_headroom(workspace.new_vertex_allowed, static_cast<std::size_t>(prepass_.pixel_count));
            launch_overlap_rejection();
        });
        emit_timeline("mark_allowed_done");
        valid_for_vertices_ = thrust::raw_pointer_cast(
            execution_.workspace.new_vertex_allowed.data()
        );
    }

    void prepare_final_allowed_mask()
    {
        CudaWorkspace& workspace = execution_.workspace;
        apply_ri_thin_filter(
            RiThinFilterRequest{
                execution_, valid_for_vertices_, prepass_.height, prepass_.width,
            }
        );
        valid_for_vertices_ = thrust::raw_pointer_cast(workspace.final_new_allowed.data());
    }

    int compute_vertex_offsets()
    {
        CudaWorkspace& workspace = execution_.workspace;
        resize_with_headroom(workspace.flags, static_cast<std::size_t>(prepass_.pixel_count));
        resize_with_headroom(workspace.offsets, static_cast<std::size_t>(prepass_.pixel_count));
        int vertex_count = 0;
        timed_cuda_stage_void(execution_.timing, "scan.vertex_prefix_sum", [&] {
            make_flags_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                valid_for_vertices_, thrust::raw_pointer_cast(workspace.flags.data()),
                prepass_.pixel_count
            );
            CUDA_CHECK(cudaGetLastError());
            vertex_count = scanned_count(workspace.flags, workspace.offsets, execution_.stream);
        });
        return vertex_count;
    }

    CudaRiFrame allocate_frame(int vertex_count)
    {
        CudaWorkspace& workspace = execution_.workspace;
        const int total_vertex_count = options_.old_vertex_count + vertex_count;
        resize_with_headroom(workspace.current_vertices_local, static_cast<std::size_t>(total_vertex_count));
        resize_with_headroom(workspace.current_vertex_rows, static_cast<std::size_t>(total_vertex_count));
        resize_with_headroom(workspace.current_vertex_cols, static_cast<std::size_t>(total_vertex_count));
        resize_with_headroom(workspace.current_vertex_ranges, static_cast<std::size_t>(total_vertex_count));
        return CudaRiFrame{
            prepass_.height, prepass_.width, options_.old_vertex_count, vertex_count,
            &workspace.current_vertices_local, &workspace.filled_vim, nullptr
        };
    }

    VertexBuildBuffers vertex_build_buffers(CudaRiFrame& frame)
    {
        CudaWorkspace& workspace = execution_.workspace;
        return VertexBuildBuffers{
            valid_for_vertices_,
            thrust::raw_pointer_cast(workspace.offsets.data()),
            pixel_xyz_,
            thrust::raw_pointer_cast(frame.vertices->data()),
            thrust::raw_pointer_cast(workspace.current_vertex_rows.data()),
            thrust::raw_pointer_cast(workspace.current_vertex_cols.data()),
            thrust::raw_pointer_cast(workspace.current_vertex_ranges.data()),
            thrust::raw_pointer_cast(frame.filled_vim->data()),
        };
    }

    void build_vertices(CudaRiFrame& frame)
    {
        timed_cuda_stage_void(execution_.timing, "scan.build_vertices", [&] {
            resize_with_headroom(*frame.filled_vim, static_cast<std::size_t>(prepass_.pixel_count));
            const VertexBuildBuffers buffers = vertex_build_buffers(frame);
            const VertexBuildLayout layout{
                {prepass_.height, prepass_.width}, prepass_.pixel_count,
                options_.old_vertex_count
            };
            build_vertices_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                buffers, layout
            );
        });
    }

    CudaPixelScanPrepass prepass_;
    const SparseMeshingConfig& config_;
    CudaExecutionContext execution_;
    VertexCompletionOptions completion_;
    ScanBuildOptions options_;
    float fov_min_{};
    float fov_range_{};
    const unsigned char* valid_for_vertices_{};
    const Vec3f* pixel_xyz_{};
};

}  // namespace

CudaRiFrame complete_scan_vertices_cuda(
    CudaPixelScanPrepass prepass,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    VertexCompletionOptions completion
)
{
    return VertexCompletionPass(prepass, config, std::move(execution), completion).run();
}

CudaRiFrame finish_pixel_vertices_from_pre_scan_cuda(
    CudaPixelScanPrepass prepass,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanBuildOptions options
)
{
    return complete_scan_vertices_cuda(
        prepass, config, execution, VertexCompletionOptions{options, true}
    );
}

}  // namespace ri_meshing::cuda_detail
