#pragma once

#include "ri_meshing/config.hpp"
#include "ri_meshing/timing.hpp"
#include "ri_meshing/types.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ri_meshing {

enum class TimingMode {
    summary,
    detailed,
    instrumented,
};

enum class FrameOverlapMode {
    none,
    pre_scan,
};

struct CudaFrameInput {
    int frame_index{};
    std::vector<Vec3f> points;
    Mat4f pose{};
};

struct StitchProxyMetrics {
    bool evaluated{};
    std::size_t proxy_segments{};
    std::size_t old_new_candidate_edges{};
    std::size_t proxy_pixels{};
    std::size_t oracle_faces{};
    std::size_t oracle_pixels{};
    std::size_t intersection_pixels{};
    std::size_t false_negative_pixels{};
    std::size_t false_positive_pixels{};
    double milliseconds{};
    double build_candidates_milliseconds{};
    double raster_proxy_milliseconds{};
    double raster_oracle_milliseconds{};
    double compare_masks_milliseconds{};
};

struct FrameProcessingStats {
    int frame_index{};
    std::size_t points_read{};
    std::size_t new_vertices{};
    std::size_t new_faces{};
    std::size_t stitch_faces{};
    std::size_t hole_faces{};
    std::size_t total_vertices{};
    std::size_t total_faces{};
    PipelineTiming timing;
    StitchProxyMetrics stitch_proxy;
};

struct MultiFrameMesh {
    std::vector<Vec3f> vertices;
    std::vector<Triangle> faces;
    std::vector<FrameProcessingStats> frame_stats;
};

using MultiFrameProgressCallback = std::function<void(const FrameProcessingStats&)>;

struct AccumDepthDebugDump {
    std::filesystem::path directory;
    std::string label;
    int max_frames{};
};

struct FilledRiDebugDump {
    std::filesystem::path directory;
    std::string label;
    int max_frames{};
};

struct OverlapDebugDump {
    std::filesystem::path directory;
    std::string label;
    int max_frames{};
};

struct RiThinFilterDebugDump {
    std::filesystem::path directory;
    std::string label;
    int max_frames{};
};

struct StitchDebugDump {
    std::filesystem::path directory;
    std::string label;
    int max_frames{};
};

struct HoleFillMetricsDump {
    std::filesystem::path directory;
    std::string label;
    int max_frames{};
};

struct StitchProxyOptions {
    bool collect_metrics{};
    bool use_blocker{true};
    bool measure_kernel_timing{};
    std::filesystem::path metrics_output_dir{};
};

struct PipelineDebugOptions {
    std::optional<AccumDepthDebugDump> accumulated_depth;
    std::optional<FilledRiDebugDump> filled_ri;
    std::optional<OverlapDebugDump> overlap;
    std::optional<RiThinFilterDebugDump> ri_thin_filter;
    std::optional<StitchDebugDump> stitching;
    std::optional<HoleFillMetricsDump> hole_filling;
};

struct CudaMultiFrameOptions {
    TimingMode timing{TimingMode::summary};
    FrameOverlapMode frame_overlap{FrameOverlapMode::pre_scan};
    StitchProxyOptions stitch_proxy;
    PipelineDebugOptions debug;
    bool overlap_timeline{};
    bool overlap_timeline_sync{};
    bool enable_hole_filling{true};
};

struct CudaPipelineWorkspace;

std::shared_ptr<CudaPipelineWorkspace> create_cuda_pipeline_workspace();

MultiFrameMesh mesh_frames_cuda(
    const std::vector<CudaFrameInput>& frames,
    const SparseMeshingConfig& config,
    const CudaMultiFrameOptions& options = {},
    CudaPipelineWorkspace* workspace = nullptr,
    MultiFrameProgressCallback progress_callback = {}
);

}  // namespace ri_meshing
