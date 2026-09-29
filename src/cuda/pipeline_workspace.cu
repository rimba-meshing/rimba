#include "pipeline_workspace.hpp"

#include "cuda_utils.hpp"

#include <exception>

namespace ri_meshing::cuda_pipeline {

void wait_prefetched_prefix_read(CudaPipelineWorkspace& workspace)
{
    FrameOverlapState& overlap = workspace.overlap;
    std::exception_ptr async_exception;
    if (overlap.prefetched_prefix_read_recorded.valid()) {
        try {
            overlap.prefetched_prefix_read_recorded.get();
        } catch (...) {
            async_exception = std::current_exception();
        }
        overlap.prefetched_prefix_read_recorded = std::shared_future<void>{};
    }
    if (overlap.has_prefetched_prefix_read_done_event) {
        if (async_exception == nullptr) {
            CUDA_CHECK(cudaEventSynchronize(overlap.prefetched_prefix_read_done_event));
        }
        CUDA_CHECK(cudaEventDestroy(overlap.prefetched_prefix_read_done_event));
        overlap.prefetched_prefix_read_done_event = nullptr;
        overlap.has_prefetched_prefix_read_done_event = false;
    }
    if (async_exception != nullptr) {
        std::rethrow_exception(async_exception);
    }
}

void wait_prefetched_pixel_scan(CudaPipelineWorkspace& workspace)
{
    if (workspace.overlap.prefetched_pixel_scan_task.valid()) {
        workspace.overlap.prefetched_pixel_scan_task.get();
    }
}

void wait_prefetched_frame_mesh(CudaPipelineWorkspace& workspace)
{
    wait_prefetched_pixel_scan(workspace);
    wait_prefetched_prefix_read(workspace);
    std::exception_ptr async_exception;
    FrameOverlapState& overlap = workspace.overlap;
    if (overlap.prefetched_frame_mesh_task.valid()) {
        try {
            overlap.prefetched_frame_mesh_task.get();
        } catch (...) {
            async_exception = std::current_exception();
        }
    }
    if (overlap.has_prefetched_frame_mesh_ready_event) {
        if (async_exception == nullptr) {
            CUDA_CHECK(cudaEventSynchronize(overlap.prefetched_frame_mesh_ready_event));
        }
        CUDA_CHECK(cudaEventDestroy(overlap.prefetched_frame_mesh_ready_event));
        overlap.prefetched_frame_mesh_ready_event = nullptr;
        overlap.has_prefetched_frame_mesh_ready_event = false;
    }
    if (async_exception != nullptr) {
        std::rethrow_exception(async_exception);
    }
}

void reset_pipeline_workspace(CudaPipelineWorkspace& workspace)
{
    wait_prefetched_frame_mesh(workspace);
    workspace.cuda.accumulated_vertices_global.clear();
    workspace.cuda.current_vertex_normal_sums.clear();
    workspace.cuda.accumulated_faces_global.clear();
    workspace.cuda.active_face_ranges.clear();
    workspace.cuda.active_vertex_ranges.clear();
    workspace.accumulated_frame_chunks.clear();
    workspace.stitch.host_boundary.clear();
    workspace.stitch.host_vertices_global.clear();
    workspace.stitch.host_current_vertex_normal_sums.clear();
    workspace.stitch.host_vertex_normal_sums_global.clear();
    workspace.stitch.filtered_old_boundary.edges.clear();
    workspace.stitch.filtered_old_boundary.old_vertex_ids.clear();
    workspace.stitch.filtered_old_boundary.new_vertex_ids.clear();
    workspace.stitch.stitch_global_to_compact.clear();
    workspace.stitch.stitch_spatial_boundary_index.clear();
    workspace.stitch.pending_boundary_vertex_changes.clear();
    workspace.stitch.pending_boundary_edges.clear();
    workspace.stitch.new_boundary_edges.clear();
    FrameOverlapState& overlap = workspace.overlap;
    overlap.prefetched_pixel_scan_task = std::future<void>{};
    overlap.has_prefetched_pixel_scan = false;
    overlap.pending_stitch_proxy_edges = StitchProxyEdgeSet{};
    overlap.has_pending_stitch_proxy = false;
    overlap.has_prefetched_frame_mesh = false;
    overlap.prefetched_prefix_read_recorded = std::shared_future<void>{};
    overlap.prefetched_prefix_read_done_event = nullptr;
    overlap.has_prefetched_prefix_read_done_event = false;
    overlap.prefetched_frame_mesh_ready_event = nullptr;
    overlap.has_prefetched_frame_mesh_ready_event = false;
    overlap.prefetched_frame_append.reset();
    overlap.prefetched_frame_append_promise.reset();
}

}  // namespace ri_meshing::cuda_pipeline

namespace ri_meshing {

std::shared_ptr<CudaPipelineWorkspace> create_cuda_pipeline_workspace()
{
    CUDA_CHECK(cudaFree(nullptr));
    return std::make_shared<CudaPipelineWorkspace>();
}

}  // namespace ri_meshing
