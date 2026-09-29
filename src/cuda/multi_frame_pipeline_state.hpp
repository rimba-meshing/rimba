#pragma once

#include "frame_overlap_scheduler.hpp"
#include "pipeline_support.hpp"
#include "stitch_append.hpp"

#include <thrust/device_vector.h>

namespace ri_meshing::cuda_pipeline {

class FrameExecution;

class CudaMultiFramePipeline {
public:
    CudaMultiFramePipeline(
        const std::vector<CudaFrameInput>& frames,
        const SparseMeshingConfig& config,
        const CudaMultiFrameOptions& options,
        CudaPipelineWorkspace& workspace,
        MultiFrameProgressCallback progress
    );

    MultiFrameMesh run();

private:
    friend class FrameExecution;

    MultiFrameMesh download_result();

    const std::vector<CudaFrameInput>& frames_;
    const SparseMeshingConfig& config_;
    const CudaMultiFrameOptions& options_;
    CudaPipelineWorkspace& workspace_;
    MultiFrameProgressCallback progress_;
    cuda_detail::CudaAccumulationCounts counts_{};
    std::vector<FrameProcessingStats> frame_stats_;
    int dumped_depth_frames_{};
    int dumped_filled_ri_frames_{};
    int dumped_overlap_frames_{};
    int dumped_ri_thin_filter_frames_{};
    bool overlap_pre_scan_{};
    OverlapTimeline timeline_;
    FrameOverlapScheduler scheduler_;
    StitchPipeline stitcher_;
};

class FrameExecution {
public:
    FrameExecution(CudaMultiFramePipeline& pipeline, std::size_t frame_position);

    FrameProcessingStats run();

private:
    friend class CudaMultiFramePipeline;

    void acquire_schedule();
    void apply_pending_boundary_edges();
    void prepare_scan_prefix();
    void prepare_accumulated_depth();
    void prepare_cuda_accumulated_depth();
    void expose_accumulated_depth();
    void expose_cuda_accumulated_depth();
    void build_vertices();
    void build_filled_pixel_vertices();
    void build_stitch_connection_mask();
    void materialize_stitch_connection_mask();
    void dump_overlap_debug();
    void dump_filled_ri();
    void dump_ri_thin_filter();
    void generate_dense_mesh();
    void append_dense_mesh();
    void extract_new_boundary();
    void register_accumulated_frame_chunk();
    void launch_stitch_normal_accumulation();
    void synchronize_stitch_inputs();
    void run_stitching();
    void finalize_accumulated_frame_chunk();
    FrameProcessingStats finish();

    CudaMultiFramePipeline& pipeline_;
    std::size_t frame_position_{};
    const CudaFrameInput& input_;
    PipelineTiming timing_;
    PipelineTiming* stage_timing_{};
    Stopwatch total_stopwatch_;
    FrameExecutionPlan plan_;
    bool use_prefetched_scan_{};
    const cuda_detail::AccumulatedDepthView* fill_support_{};
    cuda_detail::CudaWorkspace* frame_workspace_{};
    cuda_detail::StageTimelineCallback frame_timeline_;
    cuda_detail::AccumulatedDepthView depth_view_{};
    const cuda_detail::AccumulatedDepthView* accumulated_depth_{};
    cuda_detail::CudaRiFrame frame_;
    const StitchConnectionMask* connection_mask_{};
    thrust::device_vector<Triangle>* device_faces_{};
    std::size_t new_face_count_{};
    int old_vertex_count_{};
    int old_face_count_{};
    std::size_t accumulated_frame_chunk_index_{};
    bool has_accumulated_frame_chunk_{};
    StitchAppendResult stitch_result_;
};

}  // namespace ri_meshing::cuda_pipeline
