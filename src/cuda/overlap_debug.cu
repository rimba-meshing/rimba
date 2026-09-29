#include "overlap_debug.hpp"

#include "stitch_gpu_helpers.hpp"

#include <cuda_runtime.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace ri_meshing::cuda_pipeline {
namespace {

std::vector<unsigned char> copy_device_mask(const unsigned char* device, std::size_t count)
{
    std::vector<unsigned char> host(count);
    CUDA_CHECK(cudaMemcpy(
        host.data(), device, count * sizeof(unsigned char), cudaMemcpyDeviceToHost
    ));
    return host;
}

std::vector<int> copy_device_indices(const int* device, std::size_t count)
{
    std::vector<int> host(count);
    CUDA_CHECK(cudaMemcpy(
        host.data(), device, count * sizeof(int), cudaMemcpyDeviceToHost
    ));
    return host;
}

std::vector<unsigned char> depth_mask(
    cuda_detail::CudaWorkspace& workspace,
    cuda_detail::AccumulatedDepthView depth,
    int height,
    int width
)
{
    materialize_accumulated_depth_cuda(workspace, depth, height, width);
    const thrust::host_vector<float> values = workspace.accumulated_depth;
    std::vector<unsigned char> mask(values.size(), 0);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        mask[idx] = std::isfinite(values[idx]) ? 1 : 0;
    }
    return mask;
}

void write_mask(
    const std::filesystem::path& path,
    const std::vector<unsigned char>& mask,
    int width,
    int height
)
{
    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << width << ' ' << height << "\n255\n";
    for (const unsigned char value : mask) {
        const unsigned char pixel = value == 0 ? 0 : 255;
        out.write(reinterpret_cast<const char*>(&pixel), sizeof(pixel));
    }
}

std::string output_stem(const OverlapDebugDump& debug_dump, int frame_index)
{
    std::ostringstream stem;
    stem << (debug_dump.label.empty() ? "overlap" : debug_dump.label)
         << "_frame_" << std::setw(6) << std::setfill('0') << frame_index;
    return stem.str();
}

void write_metadata(
    const std::filesystem::path& path,
    const OverlapDebugRequest& request,
    const std::vector<unsigned char>& old_mask,
    const std::vector<unsigned char>& new_mask,
    const std::vector<unsigned char>& accepted,
    const std::vector<unsigned char>& rejected,
    const std::vector<unsigned char>& raw_rejected,
    const std::vector<unsigned char>& overlap_accepted,
    const std::vector<unsigned char>& thin_removed,
    const std::vector<unsigned char>& frontier
)
{
    std::ofstream out(path);
    out << "width " << request.width << '\n';
    out << "height " << request.height << '\n';
    out << "frame " << request.frame_index << '\n';
    out << "neighbor_tolerance " << request.neighbor_tolerance << '\n';
    out << "old_pixels " << std::count(old_mask.begin(), old_mask.end(), 1) << '\n';
    out << "new_candidates " << std::count(new_mask.begin(), new_mask.end(), 1) << '\n';
    out << "accepted_pixels " << std::count(accepted.begin(), accepted.end(), 1) << '\n';
    out << "effective_new_pixels " << std::count(accepted.begin(), accepted.end(), 1) << '\n';
    out << "rejected_pixels " << std::count(raw_rejected.begin(), raw_rejected.end(), 1) << '\n';
    out << "visible_rejected_pixels " << std::count(rejected.begin(), rejected.end(), 1) << '\n';
    out << "overlap_accepted_pixels " << std::count(overlap_accepted.begin(), overlap_accepted.end(), 1) << '\n';
    out << "thin_removed_pixels " << std::count(thin_removed.begin(), thin_removed.end(), 1) << '\n';
    std::size_t same_pixel_rejected = 0;
    for (std::size_t idx = 0; idx < raw_rejected.size(); ++idx) {
        same_pixel_rejected += raw_rejected[idx] != 0 && old_mask[idx] != 0;
    }
    out << "same_pixel_rejected_pixels " << same_pixel_rejected << '\n';
    out << "neighbor_rejected_pixels "
        << std::count(raw_rejected.begin(), raw_rejected.end(), 1) - same_pixel_rejected << '\n';
    out << "frontier_pixels " << std::count(frontier.begin(), frontier.end(), 1) << '\n';
}

}  // namespace

void dump_overlap_debug(OverlapDebugRequest request)
{
    const std::size_t pixel_count = static_cast<std::size_t>(request.height) * request.width;
    std::filesystem::create_directories(request.debug_dump.directory);
    const std::vector<unsigned char> old_mask = depth_mask(
        request.workspace, request.depth, request.height, request.width
    );
    const std::vector<unsigned char> raw_new_mask = copy_device_mask(
        request.new_valid, pixel_count
    );
    const std::vector<unsigned char> overlap_allowed_mask = copy_device_mask(
        request.overlap_allowed, pixel_count
    );
    const std::vector<unsigned char> final_allowed_mask = copy_device_mask(
        request.final_allowed, pixel_count
    );
    const std::vector<unsigned char> thin_removed_mask = copy_device_mask(
        request.thin_removed, pixel_count
    );
    std::vector<unsigned char> effective_new_mask(pixel_count, 0);
    if (request.effective_new_vim != nullptr) {
        const std::vector<int> filled_vim = copy_device_indices(
            request.effective_new_vim, pixel_count
        );
        for (std::size_t idx = 0; idx < pixel_count; ++idx) {
            effective_new_mask[idx] = filled_vim[idx] >= 0 ? 1 : 0;
            if (raw_new_mask[idx] != 0 && final_allowed_mask[idx] == 0) {
                effective_new_mask[idx] = 0;
            }
        }
    }
    std::vector<unsigned char> new_mask(pixel_count, 0);
    std::vector<unsigned char> accepted(pixel_count, 0);
    std::vector<unsigned char> rejected(pixel_count, 0);
    std::vector<unsigned char> raw_rejected(pixel_count, 0);
    std::vector<unsigned char> overlap_accepted(pixel_count, 0);
    std::vector<unsigned char> thin_removed(pixel_count, 0);
    std::vector<unsigned char> same_pixel_rejected(pixel_count, 0);
    std::vector<unsigned char> neighbor_rejected(pixel_count, 0);
    for (std::size_t idx = 0; idx < pixel_count; ++idx) {
        new_mask[idx] = raw_new_mask[idx] != 0;
        overlap_accepted[idx] = new_mask[idx] && overlap_allowed_mask[idx] != 0;
        if (request.effective_new_vim == nullptr) {
            effective_new_mask[idx] = new_mask[idx] && final_allowed_mask[idx] != 0;
        }
        if (old_mask[idx] != 0) {
            effective_new_mask[idx] = 0;
        }
        accepted[idx] = effective_new_mask[idx];
        raw_rejected[idx] = new_mask[idx] && !overlap_accepted[idx];
        // A propagated label does not hide an actual overlap rejection.
        rejected[idx] = raw_rejected[idx];
        thin_removed[idx] = new_mask[idx] && thin_removed_mask[idx] != 0;
        same_pixel_rejected[idx] = raw_rejected[idx] && old_mask[idx];
        neighbor_rejected[idx] = raw_rejected[idx] && !same_pixel_rejected[idx];
    }
    const thrust::host_vector<unsigned char> connection_mask_host =
        request.workspace.stitch_connection_mask;
    std::vector<unsigned char> frontier(pixel_count, 0);
    if (connection_mask_host.size() == pixel_count) {
        std::copy(connection_mask_host.begin(), connection_mask_host.end(), frontier.begin());
    }

    const std::string stem = output_stem(request.debug_dump, request.frame_index);
    write_mask(request.debug_dump.directory / (stem + "_old_mask.pgm"), old_mask, request.width, request.height);
    write_mask(request.debug_dump.directory / (stem + "_candidate_mask.pgm"), new_mask, request.width, request.height);
    write_mask(
        request.debug_dump.directory / (stem + "_overlap_accepted_mask.pgm"),
        overlap_accepted, request.width, request.height
    );
    write_mask(request.debug_dump.directory / (stem + "_accepted_mask.pgm"), accepted, request.width, request.height);
    write_mask(request.debug_dump.directory / (stem + "_rejected_mask.pgm"), rejected, request.width, request.height);
    write_mask(
        request.debug_dump.directory / (stem + "_thin_removed_mask.pgm"),
        thin_removed, request.width, request.height
    );
    write_mask(
        request.debug_dump.directory / (stem + "_same_pixel_rejected_mask.pgm"),
        same_pixel_rejected, request.width, request.height
    );
    write_mask(
        request.debug_dump.directory / (stem + "_neighbor_rejected_mask.pgm"),
        neighbor_rejected, request.width, request.height
    );
    write_mask(request.debug_dump.directory / (stem + "_frontier_mask.pgm"), frontier, request.width, request.height);
    write_metadata(
        request.debug_dump.directory / (stem + "_overlap.txt"), request,
        old_mask, new_mask, accepted, rejected, raw_rejected,
        overlap_accepted, thin_removed, frontier
    );
}

}  // namespace ri_meshing::cuda_pipeline
