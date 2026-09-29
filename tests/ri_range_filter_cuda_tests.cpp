#include "scan_internal.cuh"

#include <thrust/host_vector.h>

#include <cassert>
#include <cmath>
#include <vector>

namespace {

ri_meshing::Vec3f make_point(
    int row,
    int col,
    float range,
    const ri_meshing::SparseMeshingConfig& config
)
{
    const double phi = config.vertical_fov.min_rad
        + static_cast<double>(row) / config.ri_height
            * (config.vertical_fov.max_rad - config.vertical_fov.min_rad);
    const double theta = static_cast<double>(col) / config.ri_width
        * 2.0 * std::acos(-1.0) - std::acos(-1.0);
    const double xy = static_cast<double>(range) * std::cos(phi);
    return ri_meshing::Vec3f{
        static_cast<float>(xy * std::cos(theta)),
        static_cast<float>(xy * std::sin(theta)),
        static_cast<float>(static_cast<double>(range) * std::sin(phi)),
    };
}

int pixel_id(int row, int col, int width)
{
    return row * width + col;
}

}  // namespace

int main()
{
    ri_meshing::SparseMeshingConfig config{};
    config.ri_height = 64;
    config.ri_width = 64;
    config.max_triangle_dist = 0.30;

    const std::vector<ri_meshing::Vec3f> points{
        make_point(10, 10, 10.0F, config),
        make_point(11, 11, 10.2F, config),
        make_point(20, 20, 20.0F, config),
        make_point(30, 30, 30.0F, config),
        make_point(30, 31, 30.1F, config),
        make_point(40, 0, 40.0F, config),
        make_point(40, 63, 40.1F, config),
        make_point(50, 10, 50.0F, config),
        make_point(50, 11, 50.1F, config),
        make_point(51, 10, 50.2F, config),
        make_point(55, 20, 60.0F, config),
        make_point(55, 21, 60.1F, config),
        make_point(55, 22, 60.2F, config),
        make_point(55, 23, 60.25F, config),
    };

    ri_meshing::cuda_detail::CudaWorkspace workspace;
    ri_meshing::PipelineTiming timing;
    const ri_meshing::cuda_detail::CudaPixelScanPrepass prepass =
        ri_meshing::cuda_detail::project_scan_to_ri_cuda(
            points,
            config,
            ri_meshing::cuda_detail::CudaExecutionContext{workspace, &timing},
            ri_meshing::cuda_detail::ScanProjectionOptions{false}
        );
    assert(cudaDeviceSynchronize() == cudaSuccess);

    const thrust::host_vector<unsigned char> projected = workspace.valid;
    const thrust::host_vector<unsigned char> filtered = workspace.clean_valid;
    const auto expect_kept = [&](int row, int col) {
        const int pixel = pixel_id(row, col, prepass.width);
        assert(projected[pixel] != 0);
        assert(filtered[pixel] != 0);
    };
    const auto expect_removed = [&](int row, int col) {
        const int pixel = pixel_id(row, col, prepass.width);
        assert(projected[pixel] != 0);
        assert(filtered[pixel] == 0);
    };

    expect_removed(10, 10);
    expect_removed(11, 11);
    expect_removed(20, 20);
    expect_removed(30, 30);
    expect_removed(30, 31);
    expect_removed(40, 0);
    expect_removed(40, 63);
    expect_removed(50, 10);
    expect_removed(50, 11);
    expect_removed(51, 10);
    expect_kept(55, 20);
    expect_kept(55, 21);
    expect_kept(55, 22);
    expect_kept(55, 23);
    return 0;
}
