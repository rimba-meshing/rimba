#pragma once

#include "pipeline_support.hpp"
#include "pipeline_workspace.hpp"
#include "ri_meshing/boundary_stitching.hpp"

#include <optional>

namespace ri_meshing::cuda_pipeline {

class FrameOverlapScheduler;

struct StitchPipelineOptions {
    StitchProxyOptions proxy;
    std::optional<StitchDebugDump> debug;
    std::optional<HoleFillMetricsDump> hole_filling;
    bool enable_hole_filling{true};
};

class StitchProxyController {
public:
    explicit StitchProxyController(StitchProxyOptions options);

    [[nodiscard]] bool paths_matched(bool has_next_pose) const;
    [[nodiscard]] bool collects_metrics() const;
    [[nodiscard]] bool uses_blocker() const;
    [[nodiscard]] bool measures_kernel_timing() const;
    [[nodiscard]] const std::filesystem::path& metrics_output_dir() const;

private:
    StitchProxyOptions options_;
};

struct StitchFrameInput {
    std::size_t frame_position{};
    int frame_index{};
    int old_vertex_count{};
    int old_face_count{};
    int new_face_count{};
    Mat4f pose{};
    std::optional<Mat4f> next_pose;
    const StitchConnectionMask* connection_mask{};
    const unsigned char* connection_mask_device{};
    const float* filled_ranges_device{};
};

class StitchPipeline {
public:
    StitchPipeline(
        CudaPipelineWorkspace& workspace,
        FrameOverlapScheduler& scheduler,
        const SparseMeshingConfig& config,
        StitchPipelineOptions options,
        OverlapTimeline& timeline
    );

    StitchAppendResult process(
        const StitchFrameInput& input,
        cuda_detail::CudaAccumulationCounts& counts,
        PipelineTiming* timing
    );

private:
    CudaPipelineWorkspace& workspace_;
    FrameOverlapScheduler& scheduler_;
    const SparseMeshingConfig& config_;
    StitchPipelineOptions options_;
    StitchProxyController proxy_controller_;
    OverlapTimeline& timeline_;
    int dumped_frames_{};
    int dumped_hole_frames_{};
};

}  // namespace ri_meshing::cuda_pipeline
