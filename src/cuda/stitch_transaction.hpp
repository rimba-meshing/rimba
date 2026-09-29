#pragma once

#include "stitch_append.hpp"

#include "ri_meshing/boundary_stitching.hpp"

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace ri_meshing::cuda_pipeline {

struct StitchTransactionResources {
    CudaPipelineWorkspace& workspace;
    FrameOverlapScheduler& scheduler;
    const SparseMeshingConfig& config;
    const StitchProxyController& proxy_controller;
    const std::optional<StitchDebugDump>& debug;
    const std::optional<HoleFillMetricsDump>& hole_filling_debug;
    bool enable_hole_filling;
    OverlapTimeline& timeline;
    int& dumped_frames;
    int& dumped_hole_frames;
};

class StitchTransaction {
public:
    StitchTransaction(
        StitchTransactionResources resources,
        const StitchFrameInput& input,
        cuda_detail::CudaAccumulationCounts& counts,
        PipelineTiming* timing
    );

    StitchAppendResult execute();

private:
    using BoundaryEdges = std::vector<std::array<int, 2>>;
    using PolylineProxyCallback = std::function<void(const BoundaryEdges&)>;

    [[nodiscard]] bool should_run() const;
    [[nodiscard]] bool proxy_metrics_enabled() const;
    [[nodiscard]] bool proxy_blocker_enabled() const;
    [[nodiscard]] const Mat4f* next_pose() const;
    [[nodiscard]] const StitchDebugDump* debug_dump() const;

    void validate() const;
    void extract_boundary();
    void download_debug_faces();
    void prepare_filter_compact();
    void prepare_compact_input();
    const FilteredOldBoundaryEdges& filter_old_boundary();
    void apply_pending_boundary_changes();
    void prepare_proxy_metrics();
    void compact_filtered_input(const FilteredOldBoundaryEdges& filtered, const Mat4f& global_to_local);
    void log_compact_debug(const FilteredOldBoundaryEdges& filtered) const;

    void solve();
    void run_stitch_core();
    [[nodiscard]] PolylineProxyCallback polyline_proxy_callback();
    void publish_polyline_proxy(const BoundaryEdges& compact_edges);
    void finish_proxy_metrics();
    void log_core_debug() const;

    void update_boundary();
    void fill_local_holes();
    void append_hole_vertices(const std::vector<Vec3f>& vertices);
    void update_boundary_from_faces(const std::vector<Triangle>& faces);
    void log_boundary_debug() const;
    StitchAppendResult append_result();
    void dump_debug_snapshot();
    void wait_for_proxy();
    void append_faces();
    StitchAppendResult make_result();

    void log(const char* event, const std::string& detail = {}) const;
    void log_cuda(const char* event, const std::string& detail = {}) const;

    StitchTransactionResources resources_;
    const StitchFrameInput& input_;
    cuda_detail::CudaAccumulationCounts& counts_;
    PipelineTiming* timing_{};
    BoundaryEdges& new_boundary_edges_;
    StitchCoreResult result_;
    StitchCoreDebug debug_;
    std::vector<Triangle> new_faces_;
    std::vector<Triangle> hole_faces_;
    BoundaryEdges stitch_boundary_edges_;
    std::size_t hole_candidate_loops_{};
    std::size_t hole_accepted_loops_{};
    CompactStitchInput compact_input_;
    StitchProxyScratch proxy_scratch_;
    StitchProxyEdgeSet proxy_edges_;
    bool proxy_overlap_started_{};
};

}  // namespace ri_meshing::cuda_pipeline
