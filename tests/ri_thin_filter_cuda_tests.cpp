#include "ri_thin_filter.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cassert>
#include <vector>

namespace {

int pixel_id(int row, int col, int width)
{
    return row * width + col;
}

}  // namespace

int main()
{
    constexpr int height = 32;
    constexpr int width = 64;
    std::vector<unsigned char> host_input(height * width, 0);

    for (int row = 5; row < 12; ++row) {
        for (int col = 3; col < 10; ++col) {
            host_input[pixel_id(row, col, width)] = 1;
        }
        for (int col = 18; col < 25; ++col) {
            host_input[pixel_id(row, col, width)] = 1;
        }
    }
    for (int row = 8; row < 10; ++row) {
        for (int col = 10; col < 18; ++col) {
            host_input[pixel_id(row, col, width)] = 1;
        }
    }
    host_input[pixel_id(6, 25, width)] = 1;
    host_input[pixel_id(20, 30, width)] = 1;
    host_input[pixel_id(21, 40, width)] = 1;

    thrust::device_vector<unsigned char> input(host_input.begin(), host_input.end());
    ri_meshing::cuda_detail::CudaWorkspace workspace;
    ri_meshing::PipelineTiming timing;
    ri_meshing::cuda_detail::apply_ri_thin_filter(
        ri_meshing::cuda_detail::RiThinFilterRequest{
            ri_meshing::cuda_detail::CudaExecutionContext{workspace, &timing},
            thrust::raw_pointer_cast(input.data()), height, width,
        }
    );
    assert(cudaDeviceSynchronize() == cudaSuccess);

    const thrust::host_vector<unsigned char> source = workspace.ri_thin_input;
    const thrust::host_vector<unsigned char> eroded = workspace.ri_thin_eroded;
    const thrust::host_vector<unsigned char> grown = workspace.ri_thin_grown;
    const thrust::host_vector<unsigned char> removed = workspace.ri_thin_removed;
    const thrust::host_vector<unsigned char> final_mask = workspace.final_new_allowed;

    assert(source[pixel_id(8, 6, width)] != 0);
    assert(eroded[pixel_id(8, 6, width)] != 0);
    assert(grown[pixel_id(5, 3, width)] != 0);
    assert(grown[pixel_id(8, 10, width)] == 0);
    assert(grown[pixel_id(9, 17, width)] == 0);
    assert(removed[pixel_id(8, 10, width)] != 0);
    assert(removed[pixel_id(6, 25, width)] != 0);
    assert(removed[pixel_id(20, 30, width)] != 0);
    assert(removed[pixel_id(21, 40, width)] != 0);
    assert(final_mask[pixel_id(8, 6, width)] != 0);
    assert(final_mask[pixel_id(8, 10, width)] == 0);
    return 0;
}
