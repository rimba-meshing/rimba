#include "ri_thin_filter.hpp"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <type_traits>

namespace ri_meshing::cuda_detail {
namespace {

struct ThinFilterBuffers {
    const unsigned char* input{};
    unsigned char* eroded{};
    unsigned char* grown{};
    unsigned char* removed{};
};

struct ThinFilterDimensions {
    int height{};
    int width{};
};

static_assert(std::is_trivially_copyable_v<ThinFilterBuffers>);
static_assert(std::is_trivially_copyable_v<ThinFilterDimensions>);

__device__ int wrap_column(int col, int width)
{
    return col < 0 ? col + width : (col >= width ? col - width : col);
}

__device__ bool input_at(
    const unsigned char* input,
    int row,
    int col,
    ThinFilterDimensions dimensions
)
{
    if (row < 0 || row >= dimensions.height) {
        return false;
    }
    return input[row * dimensions.width + wrap_column(col, dimensions.width)] != 0;
}

__global__ void erode_kernel(
    const unsigned char* input,
    unsigned char* eroded,
    ThinFilterDimensions dimensions
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = dimensions.height * dimensions.width;
    if (pixel >= pixel_count) {
        return;
    }
    const int row = pixel / dimensions.width;
    const int col = pixel - row * dimensions.width;
    bool survives = input[pixel] != 0;
    for (int dr = -1; dr <= 1 && survives; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (!input_at(input, row + dr, col + dc, dimensions)) {
                survives = false;
                break;
            }
        }
    }
    eroded[pixel] = static_cast<unsigned char>(survives);
}

__global__ void grow_kernel(
    const unsigned char* input,
    const unsigned char* eroded,
    unsigned char* grown,
    ThinFilterDimensions dimensions
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = dimensions.height * dimensions.width;
    if (pixel >= pixel_count) {
        return;
    }
    const int row = pixel / dimensions.width;
    const int col = pixel - row * dimensions.width;
    bool can_grow = false;
    for (int dr = -1; dr <= 1 && !can_grow; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (input_at(eroded, row + dr, col + dc, dimensions)) {
                can_grow = true;
                break;
            }
        }
    }
    grown[pixel] = static_cast<unsigned char>(input[pixel] != 0 && can_grow);
}

__global__ void removed_kernel(
    const unsigned char* input,
    const unsigned char* grown,
    unsigned char* removed,
    ThinFilterDimensions dimensions
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = dimensions.height * dimensions.width;
    if (pixel >= pixel_count) {
        return;
    }
    removed[pixel] = static_cast<unsigned char>(input[pixel] != 0 && grown[pixel] == 0);
}

}  // namespace

void apply_ri_thin_filter(RiThinFilterRequest request)
{
    const int pixel_count = request.height * request.width;
    auto& workspace = request.execution.workspace;
    resize_with_headroom(workspace.ri_thin_input, static_cast<std::size_t>(pixel_count));
    resize_with_headroom(workspace.ri_thin_eroded, static_cast<std::size_t>(pixel_count));
    resize_with_headroom(workspace.ri_thin_grown, static_cast<std::size_t>(pixel_count));
    resize_with_headroom(workspace.ri_thin_removed, static_cast<std::size_t>(pixel_count));
    resize_with_headroom(workspace.final_new_allowed, static_cast<std::size_t>(pixel_count));

    constexpr int block = 256;
    const int grid = (pixel_count + block - 1) / block;
    const ThinFilterDimensions dimensions{request.height, request.width};
    timed_cuda_stage_void(request.execution.timing, "scan.filter_ri_thin_components", [&] {
        copy_device_to_device_async(
            thrust::raw_pointer_cast(workspace.ri_thin_input.data()),
            request.input,
            static_cast<std::size_t>(pixel_count),
            request.execution.stream
        );
        erode_kernel<<<grid, block, 0, request.execution.stream>>>(
            thrust::raw_pointer_cast(workspace.ri_thin_input.data()),
            thrust::raw_pointer_cast(workspace.ri_thin_eroded.data()),
            dimensions
        );
        grow_kernel<<<grid, block, 0, request.execution.stream>>>(
            thrust::raw_pointer_cast(workspace.ri_thin_input.data()),
            thrust::raw_pointer_cast(workspace.ri_thin_eroded.data()),
            thrust::raw_pointer_cast(workspace.ri_thin_grown.data()),
            dimensions
        );
        removed_kernel<<<grid, block, 0, request.execution.stream>>>(
            thrust::raw_pointer_cast(workspace.ri_thin_input.data()),
            thrust::raw_pointer_cast(workspace.ri_thin_grown.data()),
            thrust::raw_pointer_cast(workspace.ri_thin_removed.data()),
            dimensions
        );
        CUDA_CHECK(cudaGetLastError());
        copy_device_to_device_async(
            thrust::raw_pointer_cast(workspace.final_new_allowed.data()),
            thrust::raw_pointer_cast(workspace.ri_thin_grown.data()),
            static_cast<std::size_t>(pixel_count),
            request.execution.stream
        );
    });
}

}  // namespace ri_meshing::cuda_detail
