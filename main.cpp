#include "ri_meshing/config.hpp"
#include "ri_meshing/cuda_meshing.hpp"
#include "ri_meshing/io.hpp"
#include "ri_meshing/memory_diagnostics.hpp"
#include "ri_meshing/timing.hpp"

#include <exception>
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cstdlib>

namespace {

struct CliOptions {
    std::string dataset{"~/Datasets/local/mai_city/bin"};
    std::string sequence{"00"};
    int frame{0};
    int frame_count{1};
    std::filesystem::path output{"build/frame_000000.ply"};
    int warmup_runs{1};
    int warmup_frame_count{0};
    int measured_runs{1};
    bool use_custom_poses{false};
    ri_meshing::TimingMode timing_mode{ri_meshing::TimingMode::summary};
    ri_meshing::FrameOverlapMode frame_overlap_mode{ri_meshing::FrameOverlapMode::pre_scan};
    bool stitch_proxy_metrics{false};
    bool stitch_proxy_blocker{true};
    bool stitch_proxy_blocker_explicit{false};
    bool stitch_proxy_kernel_timing{false};
    std::optional<std::filesystem::path> stitch_proxy_metrics_dir;
    bool overlap_timeline{false};
    bool overlap_timeline_sync{false};
    bool enable_hole_filling{true};
    std::optional<double> max_triangle_dist;
    std::optional<double> accumulated_depth_cull_radius;
    std::optional<double> min_triangle_area;
    std::optional<double> min_triangle_quality;
    std::optional<double> max_hole_ri_area_pixels;
    std::optional<double> filled_pixel_macro_max_error;
    std::optional<std::filesystem::path> debug_depth_dir;
    int debug_depth_max_frames{0};
    std::optional<std::filesystem::path> debug_filled_ri_dir;
    int debug_filled_ri_max_frames{0};
    std::optional<std::filesystem::path> debug_overlap_dir;
    int debug_overlap_max_frames{0};
    std::optional<std::filesystem::path> debug_ri_thin_dir;
    int debug_ri_thin_max_frames{0};
    std::optional<std::filesystem::path> debug_stitch_dir;
    int debug_stitch_max_frames{0};
    std::optional<std::filesystem::path> debug_hole_fill_dir;
    int debug_hole_fill_max_frames{0};
};

void print_usage(std::string_view executable)
{
    std::cout << "Usage: " << executable
              << " [--dataset PATH] [--sequence 00] [--frame 0] [--output mesh.ply]"
              << " [--frame-count N] [--use-custom-poses]"
              << " [--warmup-runs 1] [--warmup-frame-count N] [--runs 1]"
              << " [--frame-overlap none|pre-scan]"
              << " [--stitch-proxy-metrics]"
              << " [--stitch-proxy-blocker|--disable-stitch-proxy-blocker]"
              << " [--stitch-proxy-kernel-timing]"
              << " [--stitch-proxy-metrics-dir DIR]"
              << " [--overlap-timeline] [--overlap-timeline-sync]"
              << " [--enable-hole-filling|--disable-hole-filling]"
              << " [--max-triangle-dist D]"
              << " [--accumulated-depth-cull-radius R]"
              << " [--min-triangle-area A] [--min-triangle-quality Q]"
              << " [--max-hole-ri-area-pixels A]"
              << " [--filled-pixel-macro-max-error E]"
              << " [--debug-depth-dir DIR] [--debug-depth-max-frames N]"
              << " [--debug-filled-ri-dir DIR] [--debug-filled-ri-max-frames N]"
              << " [--debug-overlap-dir DIR] [--debug-overlap-max-frames N]"
              << " [--debug-ri-thin-dir DIR] [--debug-ri-thin-max-frames N]"
              << " [--debug-stitch-dir DIR] [--debug-stitch-max-frames N]"
              << " [--debug-hole-fill-dir DIR] [--debug-hole-fill-max-frames N]"
              << " [--timing summary|detailed|instrumented]\n";
}

ri_meshing::TimingMode parse_timing_mode(std::string_view value)
{
    if (value == "summary") {
        return ri_meshing::TimingMode::summary;
    }
    if (value == "detailed") {
        return ri_meshing::TimingMode::detailed;
    }
    if (value == "instrumented") {
        return ri_meshing::TimingMode::instrumented;
    }
    throw std::invalid_argument("Unknown --timing value: " + std::string(value));
}

std::string_view timing_mode_name(ri_meshing::TimingMode mode)
{
    switch (mode) {
    case ri_meshing::TimingMode::summary:
        return "summary";
    case ri_meshing::TimingMode::detailed:
        return "detailed";
    case ri_meshing::TimingMode::instrumented:
        return "instrumented";
    }
    return "unknown";
}

ri_meshing::FrameOverlapMode parse_frame_overlap_mode(std::string_view value)
{
    if (value == "none") {
        return ri_meshing::FrameOverlapMode::none;
    }
    if (value == "pre-scan") {
        return ri_meshing::FrameOverlapMode::pre_scan;
    }
    throw std::invalid_argument("Unknown --frame-overlap value: " + std::string(value));
}

std::string_view frame_overlap_mode_name(ri_meshing::FrameOverlapMode mode)
{
    switch (mode) {
    case ri_meshing::FrameOverlapMode::none:
        return "none";
    case ri_meshing::FrameOverlapMode::pre_scan:
        return "pre-scan";
    }
    return "unknown";
}

void print_timing_mode(std::ostream& out, ri_meshing::TimingMode mode)
{
    out << "Timing mode: " << timing_mode_name(mode);
#if !defined(RI_MESHING_ENABLE_DETAILED_TIMING)
    if (mode != ri_meshing::TimingMode::summary) {
        out << " (stage instrumentation compiled out)";
    }
#endif
    out << '\n';
}

bool apply_boolean_option(CliOptions& options, std::string_view option)
{
    struct FlagOption {
        std::string_view name;
        bool CliOptions::*field;
        bool value;
    };
    static constexpr std::array flags{
        FlagOption{"--use-custom-poses", &CliOptions::use_custom_poses, true},
        FlagOption{"--stitch-proxy-metrics", &CliOptions::stitch_proxy_metrics, true},
        FlagOption{"--stitch-proxy-blocker", &CliOptions::stitch_proxy_blocker, true},
        FlagOption{"--disable-stitch-proxy-blocker", &CliOptions::stitch_proxy_blocker, false},
        FlagOption{"--stitch-proxy-kernel-timing", &CliOptions::stitch_proxy_kernel_timing, true},
        FlagOption{"--overlap-timeline", &CliOptions::overlap_timeline, true},
        FlagOption{"--overlap-timeline-sync", &CliOptions::overlap_timeline_sync, true},
        FlagOption{"--enable-hole-filling", &CliOptions::enable_hole_filling, true},
        FlagOption{"--disable-hole-filling", &CliOptions::enable_hole_filling, false},
    };
    const auto found = std::ranges::find(flags, option, &FlagOption::name);
    if (found == flags.end()) {
        return false;
    }
    options.*found->field = found->value;
    if (option == "--stitch-proxy-blocker" || option == "--disable-stitch-proxy-blocker") {
        options.stitch_proxy_blocker_explicit = true;
    }
    return true;
}

using ValueSetter = void (*)(CliOptions&, std::string_view);

struct ValueOption {
    std::string_view name;
    ValueSetter set;
};

const std::array<ValueOption, 29> value_options{{
    {"--dataset", +[](CliOptions& o, std::string_view v) { o.dataset = v; }},
    {"--sequence", +[](CliOptions& o, std::string_view v) { o.sequence = v; }},
    {"--frame", +[](CliOptions& o, std::string_view v) { o.frame = std::stoi(std::string(v)); }},
    {"--frame-count", +[](CliOptions& o, std::string_view v) { o.frame_count = std::stoi(std::string(v)); }},
    {"--output", +[](CliOptions& o, std::string_view v) { o.output = v; }},
    {"--warmup-runs", +[](CliOptions& o, std::string_view v) { o.warmup_runs = std::stoi(std::string(v)); }},
    {"--warmup-frame-count", +[](CliOptions& o, std::string_view v) {
        o.warmup_frame_count = std::stoi(std::string(v));
    }},
    {"--runs", +[](CliOptions& o, std::string_view v) { o.measured_runs = std::stoi(std::string(v)); }},
    {"--frame-overlap", +[](CliOptions& o, std::string_view v) { o.frame_overlap_mode = parse_frame_overlap_mode(v); }},
    {"--stitch-proxy-metrics-dir", +[](CliOptions& o, std::string_view v) { o.stitch_proxy_metrics_dir = v; }},
    {"--max-triangle-dist", +[](CliOptions& o, std::string_view v) { o.max_triangle_dist = std::stod(std::string(v)); }},
    {"--accumulated-depth-cull-radius", +[](CliOptions& o, std::string_view v) {
        o.accumulated_depth_cull_radius = std::stod(std::string(v));
    }},
    {"--min-triangle-area", +[](CliOptions& o, std::string_view v) { o.min_triangle_area = std::stod(std::string(v)); }},
    {"--min-triangle-quality", +[](CliOptions& o, std::string_view v) { o.min_triangle_quality = std::stod(std::string(v)); }},
    {"--max-hole-ri-area-pixels", +[](CliOptions& o, std::string_view v) { o.max_hole_ri_area_pixels = std::stod(std::string(v)); }},
    {"--filled-pixel-macro-max-error", +[](CliOptions& o, std::string_view v) { o.filled_pixel_macro_max_error = std::stod(std::string(v)); }},
    {"--debug-depth-dir", +[](CliOptions& o, std::string_view v) { o.debug_depth_dir = v; }},
    {"--debug-depth-max-frames", +[](CliOptions& o, std::string_view v) { o.debug_depth_max_frames = std::stoi(std::string(v)); }},
    {"--debug-filled-ri-dir", +[](CliOptions& o, std::string_view v) { o.debug_filled_ri_dir = v; }},
    {"--debug-filled-ri-max-frames", +[](CliOptions& o, std::string_view v) { o.debug_filled_ri_max_frames = std::stoi(std::string(v)); }},
    {"--debug-overlap-dir", +[](CliOptions& o, std::string_view v) { o.debug_overlap_dir = v; }},
    {"--debug-overlap-max-frames", +[](CliOptions& o, std::string_view v) { o.debug_overlap_max_frames = std::stoi(std::string(v)); }},
    {"--debug-ri-thin-dir", +[](CliOptions& o, std::string_view v) { o.debug_ri_thin_dir = v; }},
    {"--debug-ri-thin-max-frames", +[](CliOptions& o, std::string_view v) { o.debug_ri_thin_max_frames = std::stoi(std::string(v)); }},
    {"--debug-stitch-dir", +[](CliOptions& o, std::string_view v) { o.debug_stitch_dir = v; }},
    {"--debug-stitch-max-frames", +[](CliOptions& o, std::string_view v) { o.debug_stitch_max_frames = std::stoi(std::string(v)); }},
    {"--debug-hole-fill-dir", +[](CliOptions& o, std::string_view v) { o.debug_hole_fill_dir = v; }},
    {"--debug-hole-fill-max-frames", +[](CliOptions& o, std::string_view v) { o.debug_hole_fill_max_frames = std::stoi(std::string(v)); }},
    {"--timing", +[](CliOptions& o, std::string_view v) { o.timing_mode = parse_timing_mode(v); }},
}};

void apply_value_option(CliOptions& options, std::string_view option, std::string_view value)
{
    const auto found = std::ranges::find(value_options, option, &ValueOption::name);
    if (found == value_options.end()) {
        throw std::invalid_argument("Unknown argument: " + std::string(option));
    }
    found->set(options, value);
}

void validate_cli_options(const CliOptions& options)
{
    if (options.warmup_runs < 0 || options.measured_runs <= 0) {
        throw std::invalid_argument("--warmup-runs must be >= 0 and --runs must be > 0");
    }
    if (options.warmup_frame_count < 0) {
        throw std::invalid_argument("--warmup-frame-count must be >= 0");
    }
    if (options.frame_count <= 0) {
        throw std::invalid_argument("--frame-count must be > 0");
    }
    if (options.debug_depth_max_frames < 0) {
        throw std::invalid_argument("--debug-depth-max-frames must be >= 0");
    }
    if (options.debug_filled_ri_max_frames < 0) {
        throw std::invalid_argument("--debug-filled-ri-max-frames must be >= 0");
    }
    if (options.debug_overlap_max_frames < 0) {
        throw std::invalid_argument("--debug-overlap-max-frames must be >= 0");
    }
    if (options.debug_ri_thin_max_frames < 0) {
        throw std::invalid_argument("--debug-ri-thin-max-frames must be >= 0");
    }
    if (options.debug_stitch_max_frames < 0) {
        throw std::invalid_argument("--debug-stitch-max-frames must be >= 0");
    }
    if (options.debug_hole_fill_max_frames < 0) {
        throw std::invalid_argument("--debug-hole-fill-max-frames must be >= 0");
    }
    if (options.debug_stitch_dir.has_value() && options.frame_count <= 1) {
        throw std::invalid_argument("--debug-stitch-dir requires --frame-count > 1");
    }
    if (options.stitch_proxy_metrics && options.frame_count <= 1) {
        throw std::invalid_argument("--stitch-proxy-metrics requires --frame-count > 1");
    }
    if (options.stitch_proxy_blocker && options.stitch_proxy_blocker_explicit
        && options.frame_count <= 1) {
        throw std::invalid_argument("--stitch-proxy-blocker requires --frame-count > 1");
    }
    if (options.frame_overlap_mode == ri_meshing::FrameOverlapMode::pre_scan
        && options.debug_filled_ri_dir.has_value()) {
        throw std::invalid_argument("--frame-overlap pre-scan is incompatible with --debug-filled-ri-dir");
    }
    if (options.frame_overlap_mode == ri_meshing::FrameOverlapMode::pre_scan
        && options.timing_mode == ri_meshing::TimingMode::detailed) {
        throw std::invalid_argument(
            "--frame-overlap pre-scan is incompatible with --timing detailed; "
            "use --timing instrumented for non-serializing stage timing"
        );
    }
    if (options.overlap_timeline_sync && !options.overlap_timeline) {
        throw std::invalid_argument("--overlap-timeline-sync requires --overlap-timeline");
    }
#if !defined(RI_MESHING_ENABLE_DEBUG_EXPORT)
    if (options.debug_stitch_dir.has_value()) {
        throw std::invalid_argument("--debug-stitch-dir requires a build with RI_MESHING_ENABLE_DEBUG_EXPORT=ON");
    }
#endif
}

CliOptions parse_cli(int argc, char** argv)
{
    CliOptions options{};
    for (int arg = 1; arg < argc; ++arg) {
        const std::string_view option(argv[arg]);
        if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (apply_boolean_option(options, option)) {
            continue;
        }
        if (arg + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + std::string(option));
        }
        apply_value_option(options, option, argv[++arg]);
    }
    validate_cli_options(options);
    return options;
}

void accumulate_timing(ri_meshing::PipelineTiming& total, const ri_meshing::PipelineTiming& run)
{
    total.total_milliseconds += run.total_milliseconds;
    if (total.stages.empty()) {
        total.stages = run.stages;
        return;
    }
    for (std::size_t idx = 0; idx < total.stages.size(); ++idx) {
        total.stages[idx].milliseconds += run.stages[idx].milliseconds;
    }
}

std::vector<ri_meshing::CudaFrameInput> load_frame_inputs(
    const ri_meshing::SparseMeshingConfig& config,
    int first_frame,
    int frame_count
)
{
    std::vector<ri_meshing::CudaFrameInput> frames;
    frames.reserve(static_cast<std::size_t>(frame_count));
    for (int offset = 0; offset < frame_count; ++offset) {
        const int frame_index = first_frame + offset;
        const auto point_path = ri_meshing::frame_bin_path(config.dataset_path, config.sequence, frame_index);
        frames.push_back(ri_meshing::CudaFrameInput{
            frame_index,
            ri_meshing::load_kitti_bin_points(point_path),
            ri_meshing::read_pose(config.dataset_path, config.sequence, frame_index, config.use_custom_poses),
        });
    }
    return frames;
}

ri_meshing::CudaMultiFrameOptions make_cuda_multiframe_options(
    const CliOptions& options,
    ri_meshing::PipelineDebugOptions debug = {}
)
{
    return ri_meshing::CudaMultiFrameOptions{
        options.timing_mode,
        options.frame_overlap_mode,
        ri_meshing::StitchProxyOptions{
            options.stitch_proxy_metrics,
            options.stitch_proxy_blocker,
            options.stitch_proxy_kernel_timing,
            options.stitch_proxy_metrics_dir.value_or(std::filesystem::path{}),
        },
        std::move(debug),
        options.overlap_timeline,
        options.overlap_timeline_sync,
        options.enable_hole_filling,
    };
}

void print_multiframe_timing(
    const std::vector<ri_meshing::FrameProcessingStats>& total_stats,
    const std::vector<ri_meshing::FrameProcessingStats>& last_stats,
    const std::vector<double>& run_call_milliseconds,
    int runs,
    double total_call_milliseconds
)
{
    std::cout << "Measured runs: " << runs << '\n';
    for (std::size_t run = 0; run < run_call_milliseconds.size(); ++run) {
        std::cout << "Measured run " << run << " call time: "
                  << run_call_milliseconds[run] << " ms\n";
    }
    std::cout << "Multi-frame call time without dataset I/O or PLY write: "
              << total_call_milliseconds / runs << " ms\n";
    std::cout << "Multi-frame call time per frame: "
              << total_call_milliseconds / runs / static_cast<double>(total_stats.size()) << " ms\n";
    for (std::size_t idx = 0; idx < total_stats.size(); ++idx) {
        const auto& total = total_stats[idx];
        const auto& last = last_stats[idx];
        std::cout << "Frame " << total.frame_index
                  << ": " << total.timing.total_milliseconds / runs << " ms"
                  << " | points " << last.points_read
                  << " | new vertices " << last.new_vertices
                  << " | new faces " << last.new_faces
                  << " | stitch faces " << last.stitch_faces
                  << " | hole faces " << last.hole_faces
                  << " | total vertices " << last.total_vertices
                  << " | total faces " << last.total_faces << '\n';
        for (const ri_meshing::StageTiming& stage : total.timing.stages) {
            std::cout << "  " << stage.name << ": " << stage.milliseconds / runs << " ms\n";
        }
    }
}

void print_stitch_proxy_metrics(
    const std::vector<ri_meshing::FrameProcessingStats>& total_stats,
    const std::vector<ri_meshing::FrameProcessingStats>& last_stats,
    int runs
)
{
    std::size_t evaluated_frames = 0;
    std::size_t proxy_segments = 0;
    std::size_t old_new_candidate_edges = 0;
    std::size_t proxy_pixels = 0;
    std::size_t oracle_faces = 0;
    std::size_t oracle_pixels = 0;
    std::size_t intersection_pixels = 0;
    std::size_t false_negative_pixels = 0;
    std::size_t false_positive_pixels = 0;
    double total_milliseconds = 0.0;
    double build_candidates_milliseconds = 0.0;
    double raster_proxy_milliseconds = 0.0;
    double raster_oracle_milliseconds = 0.0;
    double compare_masks_milliseconds = 0.0;
    for (std::size_t idx = 0; idx < last_stats.size(); ++idx) {
        const ri_meshing::StitchProxyMetrics& last = last_stats[idx].stitch_proxy;
        if (!last.evaluated) {
            continue;
        }
        ++evaluated_frames;
        proxy_segments += last.proxy_segments;
        old_new_candidate_edges += last.old_new_candidate_edges;
        proxy_pixels += last.proxy_pixels;
        oracle_faces += last.oracle_faces;
        oracle_pixels += last.oracle_pixels;
        intersection_pixels += last.intersection_pixels;
        false_negative_pixels += last.false_negative_pixels;
        false_positive_pixels += last.false_positive_pixels;
        total_milliseconds += total_stats[idx].stitch_proxy.milliseconds / static_cast<double>(runs);
        build_candidates_milliseconds += total_stats[idx].stitch_proxy.build_candidates_milliseconds
            / static_cast<double>(runs);
        raster_proxy_milliseconds += total_stats[idx].stitch_proxy.raster_proxy_milliseconds
            / static_cast<double>(runs);
        raster_oracle_milliseconds += total_stats[idx].stitch_proxy.raster_oracle_milliseconds
            / static_cast<double>(runs);
        compare_masks_milliseconds += total_stats[idx].stitch_proxy.compare_masks_milliseconds
            / static_cast<double>(runs);
    }
    if (evaluated_frames == 0) {
        return;
    }
    const double recall = oracle_pixels == 0
        ? 1.0
        : static_cast<double>(intersection_pixels) / static_cast<double>(oracle_pixels);
    const double precision = proxy_pixels == 0
        ? 1.0
        : static_cast<double>(intersection_pixels) / static_cast<double>(proxy_pixels);
    const double f1 = precision + recall == 0.0
        ? 0.0
        : 2.0 * precision * recall / (precision + recall);
    std::cout << "Stitch proxy metrics frames: " << evaluated_frames << '\n';
    std::cout << "Stitch proxy build+compare time: " << total_milliseconds
              << " ms total | " << total_milliseconds / static_cast<double>(evaluated_frames)
              << " ms/evaluated frame\n";
    std::cout << "Stitch proxy timing split: candidates " << build_candidates_milliseconds
              << " ms | proxy raster " << raster_proxy_milliseconds
              << " ms | oracle raster " << raster_oracle_milliseconds
              << " ms | mask compare " << compare_masks_milliseconds << " ms\n";
    std::cout << "Stitch proxy segments: " << proxy_segments
              << " | old-new candidate edges " << old_new_candidate_edges << '\n';
    std::cout << "Stitch proxy pixels: " << proxy_pixels
              << " | oracle pixels " << oracle_pixels
              << " | intersection " << intersection_pixels << '\n';
    std::cout << "Stitch proxy false negatives: " << false_negative_pixels
              << " | false positives " << false_positive_pixels << '\n';
    std::cout << "Stitch proxy recall: " << recall
              << " | precision: " << precision
              << " | F1: " << f1
              << " | oracle faces: " << oracle_faces << '\n';
}

ri_meshing::SparseMeshingConfig make_config(const CliOptions& options)
{
    ri_meshing::SparseMeshingConfig config{};
    config.dataset_path = options.dataset;
    config.sequence = options.sequence;
    config.use_custom_poses = options.use_custom_poses;
    if (options.max_triangle_dist) config.max_triangle_dist = *options.max_triangle_dist;
    if (options.accumulated_depth_cull_radius) {
        config.accumulated_depth_cull_radius = *options.accumulated_depth_cull_radius;
    }
    if (options.min_triangle_area) config.min_triangle_area = *options.min_triangle_area;
    if (options.min_triangle_quality) config.min_triangle_quality = *options.min_triangle_quality;
    if (options.max_hole_ri_area_pixels) {
        config.max_hole_ri_area_pixels = *options.max_hole_ri_area_pixels;
    }
    if (options.filled_pixel_macro_max_error) {
        config.filled_pixel_macro_max_error = *options.filled_pixel_macro_max_error;
    }
    ri_meshing::validate_config(config);
    return config;
}

#if defined(RI_MESHING_HAS_CUDA)

struct MultiDebugDumps {
    std::optional<ri_meshing::AccumDepthDebugDump> depth;
    std::optional<ri_meshing::FilledRiDebugDump> filled_ri;
    std::optional<ri_meshing::OverlapDebugDump> overlap;
    std::optional<ri_meshing::RiThinFilterDebugDump> ri_thin_filter;
    std::optional<ri_meshing::StitchDebugDump> stitching;
    std::optional<ri_meshing::HoleFillMetricsDump> hole_filling;
};

MultiDebugDumps make_multi_debug_dumps(const CliOptions& options)
{
    MultiDebugDumps dumps;
    if (options.debug_depth_dir) {
        dumps.depth = ri_meshing::AccumDepthDebugDump{
            *options.debug_depth_dir,
            "cuda-fast",
            options.debug_depth_max_frames,
        };
    }
    if (options.debug_filled_ri_dir) {
        dumps.filled_ri = ri_meshing::FilledRiDebugDump{
            *options.debug_filled_ri_dir,
            "iterative-5",
            options.debug_filled_ri_max_frames,
        };
    }
    if (options.debug_overlap_dir) {
        dumps.overlap = ri_meshing::OverlapDebugDump{
            *options.debug_overlap_dir,
            "overlap",
            options.debug_overlap_max_frames,
        };
    }
    if (options.debug_ri_thin_dir) {
        dumps.ri_thin_filter = ri_meshing::RiThinFilterDebugDump{
            *options.debug_ri_thin_dir,
            "ri-thin",
            options.debug_ri_thin_max_frames,
        };
    }
#if defined(RI_MESHING_ENABLE_DEBUG_EXPORT)
    if (options.debug_stitch_dir) {
        dumps.stitching = ri_meshing::StitchDebugDump{
            *options.debug_stitch_dir, "stitch", options.debug_stitch_max_frames
        };
    }
#endif
    if (options.debug_hole_fill_dir) {
        dumps.hole_filling = ri_meshing::HoleFillMetricsDump{
            *options.debug_hole_fill_dir,
            "hole-fill",
            options.debug_hole_fill_max_frames,
        };
    }
    return dumps;
}

void accumulate_multiframe_stats(
    std::vector<ri_meshing::FrameProcessingStats>& total,
    const std::vector<ri_meshing::FrameProcessingStats>& run
)
{
    if (total.empty()) {
        total = run;
        return;
    }
    for (std::size_t idx = 0; idx < total.size(); ++idx) {
        accumulate_timing(total[idx].timing, run[idx].timing);
        auto& dst = total[idx].stitch_proxy;
        const auto& src = run[idx].stitch_proxy;
        dst.milliseconds += src.milliseconds;
        dst.build_candidates_milliseconds += src.build_candidates_milliseconds;
        dst.raster_proxy_milliseconds += src.raster_proxy_milliseconds;
        dst.raster_oracle_milliseconds += src.raster_oracle_milliseconds;
        dst.compare_masks_milliseconds += src.compare_masks_milliseconds;
    }
}

struct MultiRunResult {
    ri_meshing::MultiFrameMesh mesh;
    std::vector<ri_meshing::FrameProcessingStats> total_stats;
    std::vector<double> run_call_milliseconds;
    double total_call_milliseconds{};
    MultiDebugDumps debug;
};

MultiRunResult run_multiple_frames(
    const CliOptions& options,
    const ri_meshing::SparseMeshingConfig& config,
    ri_meshing::CudaPipelineWorkspace* workspace
)
{
    const auto warmup_options = make_cuda_multiframe_options(options);
    const int warmup_frame_count = options.warmup_frame_count == 0
        ? options.frame_count
        : options.warmup_frame_count;
    const auto warmup_frames = options.warmup_runs == 0
        ? std::vector<ri_meshing::CudaFrameInput>{}
        : load_frame_inputs(config, options.frame, warmup_frame_count);
    for (int run = 0; run < options.warmup_runs; ++run) {
        ri_meshing::set_memory_pass("warmup", run);
        static_cast<void>(
            ri_meshing::mesh_frames_cuda(warmup_frames, config, warmup_options, workspace)
        );
    }
    const auto frames = load_frame_inputs(config, options.frame, options.frame_count);
    MultiDebugDumps debug = make_multi_debug_dumps(options);
    std::optional<ri_meshing::MultiFrameMesh> mesh;
    std::vector<ri_meshing::FrameProcessingStats> total_stats;
    std::vector<double> run_call_milliseconds;
    run_call_milliseconds.reserve(static_cast<std::size_t>(options.measured_runs));
    double total_call_milliseconds = 0.0;
    for (int run = 0; run < options.measured_runs; ++run) {
        ri_meshing::set_memory_pass("measured", run);
        const bool final_run = run == options.measured_runs - 1;
        ri_meshing::PipelineDebugOptions run_debug{
            final_run ? debug.depth : std::nullopt,
            final_run ? debug.filled_ri : std::nullopt,
            final_run ? debug.overlap : std::nullopt,
            final_run ? debug.ri_thin_filter : std::nullopt,
            final_run ? debug.stitching : std::nullopt,
            final_run ? debug.hole_filling : std::nullopt,
        };
        auto run_options = make_cuda_multiframe_options(options, std::move(run_debug));
        run_options.overlap_timeline = options.overlap_timeline && final_run;
        run_options.overlap_timeline_sync = options.overlap_timeline_sync && final_run;
        const ri_meshing::Stopwatch stopwatch;
        mesh = ri_meshing::mesh_frames_cuda(frames, config, run_options, workspace);
        const double run_milliseconds = stopwatch.elapsed_milliseconds();
        run_call_milliseconds.push_back(run_milliseconds);
        total_call_milliseconds += run_milliseconds;
        accumulate_multiframe_stats(total_stats, mesh->frame_stats);
    }
    return MultiRunResult{
        std::move(*mesh), std::move(total_stats), std::move(run_call_milliseconds),
        total_call_milliseconds, std::move(debug)
    };
}

void print_multi_result(
    const CliOptions& options,
    const ri_meshing::SparseMeshingConfig& config,
    const MultiRunResult& result
)
{
    std::cout << "Frames: " << options.frame_count << " starting at " << options.frame << '\n';
    std::cout << "Vertices: " << result.mesh.vertices.size() << '\n';
    std::cout << "Faces: " << result.mesh.faces.size() << '\n';
    std::cout << "Warmup runs: " << options.warmup_runs << '\n';
    std::cout << "Warmup frame count: "
              << (options.warmup_frame_count == 0 ? options.frame_count : options.warmup_frame_count)
              << '\n';
    std::cout << "Vertex mode: filled-pixel-coarsened\n";
    std::cout << "Filled-pixel fill: iterative-5 (fixed)\n";
    std::cout << "Filled-pixel macro coarsening: up to 16x16 cells, max error "
              << config.filled_pixel_macro_max_error << " m\n";
    std::cout << "Accumulated fill support: enabled (fixed)\n";
    std::cout << "Accumulated-depth culling radius: " << config.accumulated_depth_cull_radius << " m\n";
    std::cout << "RI thin component filter: enabled (fixed)\n";
    std::cout << "Frame overlap: " << frame_overlap_mode_name(options.frame_overlap_mode) << '\n';
    std::cout << "Stitch proxy metrics: " << (options.stitch_proxy_metrics ? "enabled" : "disabled") << '\n';
    std::cout << "Stitch proxy blocker: " << (options.stitch_proxy_blocker ? "enabled" : "disabled") << '\n';
    std::cout << "Stitch proxy kernel timing: "
              << (options.stitch_proxy_kernel_timing ? "enabled" : "disabled") << '\n';
    std::cout << "Stitch proxy mode: polyline\n";
    if (options.stitch_proxy_metrics_dir) {
        std::cout << "Stitch proxy metrics maps: "
                  << options.stitch_proxy_metrics_dir->string() << '\n';
    }
    std::cout << "Overlap timeline: " << (options.overlap_timeline ? "enabled" : "disabled") << '\n';
    std::cout << "Overlap timeline sync: " << (options.overlap_timeline_sync ? "enabled" : "disabled") << '\n';
    std::cout << "Boundary stitching: enabled (fixed)\n";
    std::cout << "Stitcher: current\n";
    std::cout << "RI frontier stitching: enabled (fixed)\n";
    std::cout << "Local seam hole filling: "
              << (options.enable_hole_filling ? "enabled" : "disabled") << '\n';
    std::cout << "Hole-fill RI area limit: " << config.max_hole_ri_area_pixels << " pixel^2\n";
    std::cout << "Stitch open-side filter: enabled (fixed)\n";
#if defined(RI_MESHING_ENABLE_MEMORY_INSTRUMENTATION)
    std::cout << "Memory instrumentation: enabled (diagnostic build; scheduling forced sequential)\n";
#endif
    if (result.debug.depth) std::cout << "Accum depth debug dir: " << result.debug.depth->directory << '\n';
    if (result.debug.filled_ri) std::cout << "Filled RI debug dir: " << result.debug.filled_ri->directory << '\n';
    if (result.debug.overlap) std::cout << "Overlap debug dir: " << result.debug.overlap->directory << '\n';
    if (result.debug.stitching) std::cout << "Stitch debug dir: " << result.debug.stitching->directory << '\n';
    if (result.debug.hole_filling) {
        std::cout << "Hole-fill metrics debug dir: " << result.debug.hole_filling->directory << '\n';
    }
    print_timing_mode(std::cout, options.timing_mode);
    print_multiframe_timing(
        result.total_stats, result.mesh.frame_stats, result.run_call_milliseconds,
        options.measured_runs, result.total_call_milliseconds
    );
    print_stitch_proxy_metrics(result.total_stats, result.mesh.frame_stats, options.measured_runs);
    std::cout << "Wrote: " << options.output << '\n';
}

#endif

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_cli(argc, argv);
        CliOptions effective_options = options;
#if defined(RI_MESHING_ENABLE_MEMORY_INSTRUMENTATION)
        // Memory snapshots inspect vector metadata at phase boundaries.  The
        // asynchronous scheduler can resize the prefetch workspace from a
        // worker thread, so this diagnostic-only binary deliberately uses a
        // sequential schedule to keep those observations race-free.
        effective_options.frame_overlap_mode = ri_meshing::FrameOverlapMode::none;
        effective_options.stitch_proxy_blocker = false;
#endif
        const ri_meshing::SparseMeshingConfig config = make_config(effective_options);

#if defined(RI_MESHING_HAS_CUDA)
        const auto workspace = ri_meshing::create_cuda_pipeline_workspace();
        const MultiRunResult result = run_multiple_frames(effective_options, config, workspace.get());
        ri_meshing::write_ascii_ply(effective_options.output, result.mesh.vertices, result.mesh.faces);
        print_multi_result(effective_options, config, result);
#else
        std::cerr << "This executable was built without CUDA. Reconfigure with cmake --preset cuda.\n";
        return 1;
#endif
    } catch (const std::exception& error) {
        std::cerr << "ri_meshing failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
