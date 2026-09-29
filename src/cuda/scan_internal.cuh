#pragma once

#include "cuda_workflow.hpp"

#include <type_traits>
#include <vector>

namespace ri_meshing::cuda_detail {

struct RiDimensions {
    int height{};
    int width{};
};

template <typename Scalar>
struct RiProjection {
    RiDimensions dimensions;
    Scalar fov_min{};
    Scalar fov_range{};
};

struct ScanProjectionOptions {
    bool allocate_fill_buffers{};
};

struct VertexCompletionOptions {
    ScanBuildOptions scan;
    bool emit_timeline{};
};

static_assert(std::is_trivially_copyable_v<RiDimensions>);
static_assert(std::is_trivially_copyable_v<RiProjection<float>>);
static_assert(std::is_trivially_copyable_v<ScanProjectionOptions>);
static_assert(std::is_trivially_copyable_v<VertexCompletionOptions>);

CudaPixelScanPrepass project_scan_to_ri_cuda(
    const std::vector<Vec3f>& points,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanProjectionOptions options
);

void fill_scan_range_image_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    CudaPixelScanPrepass prepass,
    const ScanBuildOptions& options
);

CudaRiFrame complete_scan_vertices_cuda(
    CudaPixelScanPrepass prepass,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    VertexCompletionOptions options
);

}  // namespace ri_meshing::cuda_detail
