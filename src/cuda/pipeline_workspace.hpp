#pragma once

#include "cuda_workflow.hpp"
#include "stitch_state.hpp"
#include "ri_meshing/boundary_graph.hpp"

#include <cuda_runtime.h>

#include <array>
#include <future>
#include <memory>
#include <vector>

namespace ri_meshing {

struct StitchWorkspace {
    BoundaryGraph host_boundary;
    std::vector<Vec3f> host_vertices_global;
    std::vector<Vec3f> host_current_vertex_normal_sums;
    std::vector<Vec3f> host_vertex_normal_sums_global;
    FilteredOldBoundaryEdges filtered_old_boundary;
    std::vector<int> stitch_global_to_compact;
    StitchSpatialBoundaryIndex stitch_spatial_boundary_index;
    std::vector<int> pending_boundary_vertex_changes;
    std::vector<std::array<int, 2>> pending_boundary_edges;
    std::vector<std::array<int, 2>> new_boundary_edges;
    BoundaryPathExtractionWorkspace stitch_path_workspace;
    CandidateOwnershipWorkspace candidate_ownership_workspace;
};

struct CudaPrefetchWorkspace {
    cuda_detail::CudaWorkspace cuda;
    cuda_detail::CudaPixelScanPrepass pixel_scan;
    cuda_detail::CudaRiFrame frame_mesh;
};

struct PrefetchedFrameAppend {
    cuda_detail::CudaAccumulationCounts counts{};
    int first_vertex{};
    int vertex_count{};
    int first_face{};
    int face_count{};
};

struct FrameOverlapState {
    std::future<void> prefetched_pixel_scan_task;
    int prefetched_frame_index{};
    bool has_prefetched_pixel_scan{};
    StitchProxyEdgeSet pending_stitch_proxy_edges;
    int pending_stitch_proxy_frame_index{};
    int pending_stitch_proxy_base_face_count{};
    bool has_pending_stitch_proxy{};
    std::future<void> prefetched_frame_mesh_task;
    int prefetched_frame_mesh_index{};
    bool has_prefetched_frame_mesh{};
    std::shared_future<void> prefetched_prefix_read_recorded;
    cudaEvent_t prefetched_prefix_read_done_event{};
    bool has_prefetched_prefix_read_done_event{};
    cudaEvent_t prefetched_frame_mesh_ready_event{};
    bool has_prefetched_frame_mesh_ready_event{};
    std::shared_ptr<PrefetchedFrameAppend> prefetched_frame_append;
    std::shared_ptr<std::promise<void>> prefetched_frame_append_promise;
};

struct CudaPipelineWorkspace {
    cuda_detail::CudaWorkspace cuda;
    CudaPrefetchWorkspace prefetch;
    StitchWorkspace stitch;
    FrameOverlapState overlap;
    std::vector<cuda_detail::AccumulatedFrameChunk> accumulated_frame_chunks;
};

namespace cuda_pipeline {

void wait_prefetched_prefix_read(CudaPipelineWorkspace& workspace);
void wait_prefetched_pixel_scan(CudaPipelineWorkspace& workspace);
void wait_prefetched_frame_mesh(CudaPipelineWorkspace& workspace);
void reset_pipeline_workspace(CudaPipelineWorkspace& workspace);

}  // namespace cuda_pipeline
}  // namespace ri_meshing
