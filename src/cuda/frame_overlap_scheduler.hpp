#pragma once

#include "pipeline_workspace.hpp"

namespace ri_meshing::cuda_pipeline {

class OverlapTimeline;

struct OverlapSchedulerResources {
    CudaPipelineWorkspace& workspace;
    const std::vector<CudaFrameInput>& frames;
    const SparseMeshingConfig& config;
    OverlapTimeline* timeline{};
};

struct OverlapSchedulerSettings {
    bool overlap_pre_scan{};
    bool stitch_proxy_blocker{};
    bool stitch_proxy_kernel_timing{};
};

struct FrameExecutionPlan {
    bool uses_prefetched_mesh{};
    bool uses_pending_proxy{};
    cuda_detail::CudaAccumulationCounts depth_counts{};
};

struct ProxyPrefetchRequest {
    std::size_t frame_position{};
    cuda_detail::CudaAccumulationCounts counts{};
    StitchProxyEdgeSet proxy_edges;
    int proxy_base_face_count{};
};

struct ProxyPrefetchJob {
    cuda_detail::CudaAccumulationCounts depth_counts;
    std::vector<cuda_detail::AccumulatedFrameChunk> frame_chunks;
    Mat4f pose;
    StitchProxyEdgeSet proxy_edges;
    Mat4f source_pose;
    int source_frame_index{};
    int frame_index{};
    bool measure_kernel_timing{};
    cudaEvent_t prefix_event{};
    cudaEvent_t ready_event{};
    std::shared_ptr<std::promise<void>> prefix_recorded;
    std::shared_ptr<PrefetchedFrameAppend> frame_append;
    std::shared_ptr<std::promise<void>> frame_append_promise;
    std::shared_future<void> frame_append_ready;
};

class FrameOverlapScheduler {
public:
    FrameOverlapScheduler(OverlapSchedulerResources resources, OverlapSchedulerSettings settings);

    FrameExecutionPlan begin_frame(int frame_index, cuda_detail::CudaAccumulationCounts counts);
    void rebase_prefetched_mesh(int new_old_vertex_count, PipelineTiming* timing);
    void paint_pending_proxy(
        const FrameExecutionPlan& plan,
        int frame_index,
        const Mat4f& pose,
        PipelineTiming* timing
    );
    [[nodiscard]] bool has_accumulated_depth(const FrameExecutionPlan& plan) const;
    [[nodiscard]] bool uses_prefetched_pixel_scan(const FrameExecutionPlan& plan, int frame_index) const;
    void wait_for_pre_scan(bool uses_prefetched_scan);
    cuda_detail::CudaWorkspace& workspace_for(const FrameExecutionPlan& plan, bool uses_prefetched_scan);
    void consume_mesh_inputs(const FrameExecutionPlan& plan, bool uses_prefetched_scan);
    void start_pre_scan(std::size_t frame_position);
    [[nodiscard]] bool supports_proxy_prefetch() const noexcept;
    bool start_proxy_prefetch(ProxyPrefetchRequest request);
    void wait_for_proxy_prefix();
    void publish_proxy(StitchAppendResult& result, std::size_t frame_position, bool consumed_pending_proxy);
    void publish_prefetched_frame_append(PrefetchedFrameAppend append);

private:
    void run_proxy_prefetch(ProxyPrefetchJob job);

    CudaPipelineWorkspace& workspace_;
    const std::vector<CudaFrameInput>& frames_;
    const SparseMeshingConfig& config_;
    bool overlap_pre_scan_{};
    bool stitch_proxy_blocker_{};
    bool stitch_proxy_kernel_timing_{};
    OverlapTimeline* timeline_{};
};

}  // namespace ri_meshing::cuda_pipeline
