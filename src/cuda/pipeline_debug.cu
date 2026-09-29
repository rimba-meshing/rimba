#include "pipeline_debug.hpp"

#include "stitch_gpu_helpers.hpp"

#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ri_meshing::cuda_pipeline {
namespace {

struct RangeSnapshot {
    thrust::host_vector<float> ranges;
    std::vector<std::uint8_t> mask;
    std::size_t valid_count{};
    float min_value{std::numeric_limits<float>::infinity()};
    float max_value{};
};

struct FilledRiDumpPaths {
    std::filesystem::path scan_mask;
    std::filesystem::path scan_ranges;
    std::filesystem::path ri_outlier_mask;
    std::filesystem::path filled_mask;
    std::filesystem::path filled_ranges;
    std::filesystem::path metadata;
};

struct FilledRiDumpMetadata {
    int height{};
    int width{};
    int frame_index{};
    const std::string& label;
    const RangeSnapshot& scan;
    const RangeSnapshot& filled;
    std::size_t ri_outlier_count{};
};

std::string ri_thin_stem(const RiThinFilterDebugDump& debug_dump, int frame_index)
{
    std::ostringstream stem;
    stem << (debug_dump.label.empty() ? "ri-thin" : debug_dump.label)
         << "_frame_" << std::setw(6) << std::setfill('0') << frame_index;
    return stem.str();
}

RangeSnapshot analyze_ranges(thrust::host_vector<float> ranges)
{
    RangeSnapshot snapshot{std::move(ranges), {}};
    snapshot.mask.resize(snapshot.ranges.size());
    for (std::size_t idx = 0; idx < snapshot.ranges.size(); ++idx) {
        const float value = snapshot.ranges[idx];
        const bool valid = std::isfinite(value) && value != 0.0F;
        snapshot.mask[idx] = valid ? 255 : 0;
        if (valid) {
            ++snapshot.valid_count;
            snapshot.min_value = std::min(snapshot.min_value, value);
            snapshot.max_value = std::max(snapshot.max_value, value);
        }
    }
    return snapshot;
}

FilledRiDumpPaths filled_ri_dump_paths(
    const FilledRiDebugDump& debug_dump,
    const std::string& label,
    int frame_index
)
{
    std::ostringstream stem;
    stem << label << "_frame_" << std::setw(6) << std::setfill('0') << frame_index;
    return FilledRiDumpPaths{
        debug_dump.directory / (stem.str() + "_scan_mask.pgm"),
        debug_dump.directory / (stem.str() + "_scan_range.f32"),
        debug_dump.directory / (stem.str() + "_ri_outlier_mask.pgm"),
        debug_dump.directory / (stem.str() + "_mask.pgm"),
        debug_dump.directory / (stem.str() + "_range.f32"),
        debug_dump.directory / (stem.str() + "_range.txt"),
    };
}

void write_mask_pgm(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& mask,
    int width,
    int height
)
{
    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(mask.data()), static_cast<std::streamsize>(mask.size()));
}

void write_ranges_raw(const std::filesystem::path& path, const thrust::host_vector<float>& ranges)
{
    std::ofstream out(path, std::ios::binary);
    out.write(
        reinterpret_cast<const char*>(ranges.data()),
        static_cast<std::streamsize>(ranges.size() * sizeof(float))
    );
}

void write_filled_ri_metadata(const std::filesystem::path& path, const FilledRiDumpMetadata& metadata)
{
    std::ofstream out(path);
    out << "width " << metadata.width << '\n';
    out << "height " << metadata.height << '\n';
    out << "frame " << metadata.frame_index << '\n';
    out << "label " << metadata.label << '\n';
    out << "scan_valid_pixels " << metadata.scan.valid_count << '\n';
    out << "scan_min_range "
        << (metadata.scan.valid_count == 0 ? 0.0F : metadata.scan.min_value) << '\n';
    out << "scan_max_range "
        << (metadata.scan.valid_count == 0 ? 0.0F : metadata.scan.max_value) << '\n';
    out << "ri_outlier_pixels " << metadata.ri_outlier_count << '\n';
    out << "valid_pixels " << metadata.filled.valid_count << '\n';
    out << "min_range "
        << (metadata.filled.valid_count == 0 ? 0.0F : metadata.filled.min_value) << '\n';
    out << "max_range "
        << (metadata.filled.valid_count == 0 ? 0.0F : metadata.filled.max_value) << '\n';
}

}  // namespace

void dump_accumulated_depth(
    cuda_detail::CudaWorkspace& workspace,
    const AccumulatedDepthDumpRequest& request
)
{
    const cuda_detail::AccumulatedDepthView depth = request.depth;
    const int height = request.height;
    const int width = request.width;
    const int frame_index = request.frame_index;
    const AccumDepthDebugDump& debug_dump = request.debug_dump;
    std::filesystem::create_directories(debug_dump.directory);
    materialize_accumulated_depth_cuda(workspace, depth, height, width);
    thrust::host_vector<float> h_depth = workspace.accumulated_depth;
    std::vector<std::uint8_t> mask(h_depth.size());
    std::size_t valid_count = 0;
    float min_depth = std::numeric_limits<float>::infinity();
    float max_depth = 0.0F;
    for (std::size_t idx = 0; idx < h_depth.size(); ++idx) {
        const float value = h_depth[idx];
        const bool valid = std::isfinite(value);
        mask[idx] = valid ? 255 : 0;
        if (valid) {
            ++valid_count;
            min_depth = std::min(min_depth, value);
            max_depth = std::max(max_depth, value);
        }
    }

    const std::string label = debug_dump.label.empty() ? "cuda-fast" : debug_dump.label;
    std::ostringstream stem;
    stem << label << "_frame_" << std::setw(6) << std::setfill('0') << frame_index;
    const std::filesystem::path pgm_path = debug_dump.directory / (stem.str() + "_mask.pgm");
    const std::filesystem::path raw_path = debug_dump.directory / (stem.str() + "_depth.f32");
    const std::filesystem::path meta_path = debug_dump.directory / (stem.str() + "_depth.txt");

    {
        std::ofstream out(pgm_path, std::ios::binary);
        out << "P5\n" << width << ' ' << height << "\n255\n";
        out.write(reinterpret_cast<const char*>(mask.data()), static_cast<std::streamsize>(mask.size()));
    }
    {
        std::ofstream out(raw_path, std::ios::binary);
        out.write(
            reinterpret_cast<const char*>(h_depth.data()),
            static_cast<std::streamsize>(h_depth.size() * sizeof(float))
        );
    }
    {
        std::ofstream out(meta_path);
        out << "width " << width << '\n';
        out << "height " << height << '\n';
        out << "frame " << frame_index << '\n';
        out << "label " << label << '\n';
        out << "valid_pixels " << valid_count << '\n';
        out << "min_depth " << (valid_count == 0 ? 0.0F : min_depth) << '\n';
        out << "max_depth " << (valid_count == 0 ? 0.0F : max_depth) << '\n';
    }
}

void dump_filled_ri(
    cuda_detail::CudaWorkspace& workspace,
    int height,
    int width,
    int frame_index,
    const FilledRiDebugDump& debug_dump
)
{
    std::filesystem::create_directories(debug_dump.directory);
    const RangeSnapshot scan = analyze_ranges(thrust::host_vector<float>{workspace.scan_ranges});
    const RangeSnapshot filled = analyze_ranges(thrust::host_vector<float>{workspace.filled_ranges});
    const thrust::host_vector<unsigned char> clean_valid = workspace.clean_valid;
    std::vector<std::uint8_t> ri_outlier_mask(scan.ranges.size());
    std::size_t ri_outlier_count = 0;
    for (std::size_t idx = 0; idx < ri_outlier_mask.size(); ++idx) {
        const bool outlier = scan.mask[idx] != 0 && clean_valid[idx] == 0;
        ri_outlier_mask[idx] = outlier ? 255 : 0;
        ri_outlier_count += outlier ? 1 : 0;
    }
    const std::string label = debug_dump.label.empty() ? "filled-ri" : debug_dump.label;
    const FilledRiDumpPaths paths = filled_ri_dump_paths(debug_dump, label, frame_index);
    write_mask_pgm(paths.scan_mask, scan.mask, width, height);
    write_ranges_raw(paths.scan_ranges, scan.ranges);
    write_mask_pgm(paths.ri_outlier_mask, ri_outlier_mask, width, height);
    write_mask_pgm(paths.filled_mask, filled.mask, width, height);
    write_ranges_raw(paths.filled_ranges, filled.ranges);
    write_filled_ri_metadata(
        paths.metadata,
        FilledRiDumpMetadata{
            height,
            width,
            frame_index,
            label,
            scan,
            filled,
            ri_outlier_count,
        }
    );
}

void dump_ri_thin_filter(
    cuda_detail::CudaWorkspace& workspace,
    int height,
    int width,
    int frame_index,
    const RiThinFilterDebugDump& debug_dump
)
{
    std::filesystem::create_directories(debug_dump.directory);
    const std::size_t pixel_count = static_cast<std::size_t>(height) * width;
    const thrust::host_vector<unsigned char> source = workspace.ri_thin_input;
    const thrust::host_vector<unsigned char> eroded = workspace.ri_thin_eroded;
    const thrust::host_vector<unsigned char> grown = workspace.ri_thin_grown;
    const thrust::host_vector<unsigned char> removed = workspace.ri_thin_removed;
    const thrust::host_vector<unsigned char> final_allowed = workspace.final_new_allowed;
    if (source.size() != pixel_count || eroded.size() != pixel_count
        || grown.size() != pixel_count || removed.size() != pixel_count
        || final_allowed.size() != pixel_count) {
        throw std::runtime_error("RI thin-filter debug buffers have an unexpected size");
    }
    const std::string stem = ri_thin_stem(debug_dump, frame_index);
    const auto write_device_mask = [&](const std::string& suffix, const auto& values) {
        std::vector<std::uint8_t> mask(values.begin(), values.end());
        write_mask_pgm(debug_dump.directory / (stem + suffix), mask, width, height);
    };
    write_device_mask("_source_mask.pgm", source);
    write_device_mask("_eroded_core_mask.pgm", eroded);
    write_device_mask("_grown_mask.pgm", grown);
    write_device_mask("_removed_mask.pgm", removed);
    write_device_mask("_final_mask.pgm", final_allowed);

    std::ofstream metadata(debug_dump.directory / (stem + "_ri_thin.txt"));
    metadata << "width " << width << '\n';
    metadata << "height " << height << '\n';
    metadata << "frame " << frame_index << '\n';
    metadata << "source_pixels " << std::count(source.begin(), source.end(), 1) << '\n';
    metadata << "eroded_core_pixels " << std::count(eroded.begin(), eroded.end(), 1) << '\n';
    metadata << "grown_pixels " << std::count(grown.begin(), grown.end(), 1) << '\n';
    metadata << "removed_pixels " << std::count(removed.begin(), removed.end(), 1) << '\n';
    metadata << "final_pixels " << std::count(final_allowed.begin(), final_allowed.end(), 1) << '\n';
    metadata << "horizontal_wrap " << 1 << '\n';
    metadata << "structuring_element " << "full_3x3" << '\n';
}

}  // namespace ri_meshing::cuda_pipeline
