#include "scan_internal.cuh"

namespace ri_meshing::cuda_detail {

CudaPixelScanPrepass scan_pixel_vertices_pre_blocker_cuda(
    const std::vector<Vec3f>& points,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanBuildOptions options
)
{
    if (execution.timeline) {
        execution.timeline("pre_scan_start");
    }
    const CudaPixelScanPrepass prepass = project_scan_to_ri_cuda(
        points, config, execution, ScanProjectionOptions{true}
    );
    fill_scan_range_image_cuda(config, execution, prepass, options);
    if (execution.timeline) {
        execution.timeline("pre_scan_done");
    }
    return prepass;
}

CudaRiFrame scan_and_build_pixel_vertices_cuda(
    const std::vector<Vec3f>& points,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanBuildOptions options
)
{
    const CudaPixelScanPrepass prepass =
        scan_pixel_vertices_pre_blocker_cuda(points, config, execution, options);
    return finish_pixel_vertices_from_pre_scan_cuda(prepass, config, execution, options);
}

}  // namespace ri_meshing::cuda_detail
