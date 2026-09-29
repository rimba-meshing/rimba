#include "multi_frame_pipeline_state.hpp"

#include "cuda_utils.hpp"
#include "pipeline_workspace.hpp"

#include <thrust/host_vector.h>

#include <algorithm>
#include <utility>

namespace ri_meshing::cuda_pipeline {
namespace {

bool enables_pre_scan_overlap(const CudaMultiFrameOptions& options)
{
    return options.frame_overlap == FrameOverlapMode::pre_scan
        && !options.debug.filled_ri
        && !options.debug.ri_thin_filter
        && options.timing != TimingMode::detailed;
}

void reserve_prefetch_workspace(
    CudaPipelineWorkspace& workspace,
    const std::vector<CudaFrameInput>& frames,
    const SparseMeshingConfig& config
)
{
    std::size_t max_point_count = 0;
    for (const CudaFrameInput& frame : frames) {
        max_point_count = std::max(max_point_count, frame.points.size());
    }
    const std::size_t pixel_count = config.ri_height * config.ri_width;
    workspace.prefetch.cuda.points.resize(max_point_count);
    workspace.prefetch.cuda.pixel_last_index.resize(pixel_count);
    workspace.prefetch.cuda.valid.resize(pixel_count);
    workspace.prefetch.cuda.clean_valid.resize(pixel_count);
    workspace.prefetch.cuda.pixel_xyz.resize(pixel_count);
    workspace.prefetch.cuda.scan_ranges.resize(pixel_count);
    workspace.prefetch.cuda.filled_ranges.resize(pixel_count);
    workspace.prefetch.cuda.fill_scratch_ranges.resize(pixel_count);
    workspace.prefetch.cuda.filled_pixel_xyz.resize(pixel_count);
    workspace.prefetch.cuda.fill_support_ranges.resize(pixel_count);
    workspace.prefetch.cuda.fill_anchor.resize(pixel_count);
    workspace.prefetch.cuda.fill_scratch_anchor.resize(pixel_count);
    const std::size_t cell_count = (config.ri_height - 1) * config.ri_width;
    workspace.prefetch.cuda.macro_quads.resize(cell_count);
    workspace.prefetch.cuda.macro_quad_count.resize(1);
    workspace.prefetch.cuda.macro_candidate_flags.resize(cell_count);
    workspace.prefetch.cuda.macro_selected_size.resize(cell_count);
    workspace.prefetch.cuda.macro_next_size.resize(cell_count);
    workspace.prefetch.cuda.quad_face_flags.resize(cell_count);
}

}  // namespace

CudaMultiFramePipeline::CudaMultiFramePipeline(
    const std::vector<CudaFrameInput>& frames,
    const SparseMeshingConfig& config,
    const CudaMultiFrameOptions& options,
    CudaPipelineWorkspace& workspace,
    MultiFrameProgressCallback progress
)
    : frames_(frames)
    , config_(config)
    , options_(options)
    , workspace_(workspace)
    , progress_(std::move(progress))
    , overlap_pre_scan_(enables_pre_scan_overlap(options))
    , timeline_(options.overlap_timeline, options.overlap_timeline_sync)
    , scheduler_(
        OverlapSchedulerResources{workspace, frames, config, &timeline_},
        OverlapSchedulerSettings{
            overlap_pre_scan_,
            options.stitch_proxy.use_blocker,
            options.stitch_proxy.measure_kernel_timing,
        }
    )
    , stitcher_(
        workspace,
        scheduler_,
        config,
        StitchPipelineOptions{
            options.stitch_proxy,
            options.debug.stitching,
            options.debug.hole_filling,
            options.enable_hole_filling,
        },
        timeline_
    )
{
    frame_stats_.reserve(frames.size());
}

MultiFrameMesh CudaMultiFramePipeline::run()
{
    reset_pipeline_workspace(workspace_);
    if (overlap_pre_scan_) {
        reserve_prefetch_workspace(
            workspace_, frames_, config_
        );
    }
    for (std::size_t frame_position = 0; frame_position < frames_.size(); ++frame_position) {
        FrameProcessingStats stats = FrameExecution(*this, frame_position).run();
        if (progress_) {
            progress_(stats);
        }
        frame_stats_.push_back(std::move(stats));
    }
    return download_result();
}

MultiFrameMesh CudaMultiFramePipeline::download_result()
{
    thrust::host_vector<Vec3f> host_vertices = workspace_.cuda.accumulated_vertices_global;
    thrust::host_vector<Triangle> host_faces = workspace_.cuda.accumulated_faces_global;
    return MultiFrameMesh{
        std::vector<Vec3f>(host_vertices.begin(), host_vertices.end()),
        std::vector<Triangle>(host_faces.begin(), host_faces.end()),
        std::move(frame_stats_),
    };
}

}  // namespace ri_meshing::cuda_pipeline

namespace ri_meshing {

MultiFrameMesh mesh_frames_cuda(
    const std::vector<CudaFrameInput>& frames,
    const SparseMeshingConfig& config,
    const CudaMultiFrameOptions& options,
    CudaPipelineWorkspace* external_workspace,
    MultiFrameProgressCallback progress_callback
)
{
    CudaPipelineWorkspace local_workspace;
    CudaPipelineWorkspace& workspace = external_workspace == nullptr
        ? local_workspace : *external_workspace;
    return cuda_pipeline::CudaMultiFramePipeline(
        frames, config, options, workspace, std::move(progress_callback)
    ).run();
}

}  // namespace ri_meshing
