#include "stitch_append.hpp"

#include "frame_overlap_scheduler.hpp"
#include "stitch_transaction.hpp"

#include <utility>

namespace ri_meshing::cuda_pipeline {

StitchProxyController::StitchProxyController(StitchProxyOptions options)
    : options_(options)
{
}

bool StitchProxyController::paths_matched(bool has_next_pose) const
{
    return has_next_pose && options_.use_blocker;
}

bool StitchProxyController::collects_metrics() const
{
    return options_.collect_metrics;
}

bool StitchProxyController::uses_blocker() const
{
    return options_.use_blocker;
}

bool StitchProxyController::measures_kernel_timing() const
{
    return options_.measure_kernel_timing;
}

const std::filesystem::path& StitchProxyController::metrics_output_dir() const
{
    return options_.metrics_output_dir;
}

StitchPipeline::StitchPipeline(
    CudaPipelineWorkspace& workspace,
    FrameOverlapScheduler& scheduler,
    const SparseMeshingConfig& config,
    StitchPipelineOptions options,
    OverlapTimeline& timeline
)
    : workspace_(workspace)
    , scheduler_(scheduler)
    , config_(config)
    , options_(std::move(options))
    , proxy_controller_(options_.proxy)
    , timeline_(timeline)
{
}

StitchAppendResult StitchPipeline::process(
    const StitchFrameInput& input,
    cuda_detail::CudaAccumulationCounts& counts,
    PipelineTiming* timing
)
{
    StitchTransaction transaction{
        StitchTransactionResources{
            workspace_,
            scheduler_,
            config_,
            proxy_controller_,
            options_.debug,
            options_.hole_filling,
            options_.enable_hole_filling,
            timeline_,
            dumped_frames_,
            dumped_hole_frames_
        },
        input,
        counts,
        timing,
    };
    return transaction.execute();
}

}  // namespace ri_meshing::cuda_pipeline
