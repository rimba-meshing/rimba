#pragma once

#include "cuda_workflow.hpp"
#include "stitch_state.hpp"

namespace ri_meshing::cuda_pipeline {

struct ProxyPaintContext {
    const SparseMeshingConfig& config;
    PipelineTiming* timing{};
    cudaStream_t stream{};
    bool measure_kernel_timing{};
    int frame_index{-1};
};

std::vector<std::array<int, 2>> batch_boundary_edges_sorted_cuda(
    cuda_detail::CudaWorkspace& workspace,
    int old_face_count,
    int new_face_count
);
void batch_boundary_edges_sorted_cuda(
    cuda_detail::CudaWorkspace& workspace,
    int old_face_count,
    int new_face_count,
    std::vector<std::array<int, 2>>& output
);
void paint_stitch_proxy_edges_into_accumulated_cuda(
    cuda_detail::CudaWorkspace& workspace,
    const StitchProxyEdgeSet& proxy_edges,
    const Mat4f& pose,
    ProxyPaintContext context
);
void materialize_accumulated_depth_cuda(
    cuda_detail::CudaWorkspace& workspace,
    cuda_detail::AccumulatedDepthView depth,
    int height,
    int width
);

}  // namespace ri_meshing::cuda_pipeline
