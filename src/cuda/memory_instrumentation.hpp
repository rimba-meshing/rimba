#pragma once

#include "ri_meshing/memory_diagnostics.hpp"
#include "pipeline_workspace.hpp"

#include <string_view>

#if defined(RI_MESHING_ENABLE_MEMORY_INSTRUMENTATION)

#include <cuda_runtime.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace ri_meshing::cuda_pipeline {
namespace memory_detail {

struct ByteCount {
    std::size_t live{};
    std::size_t capacity{};
};

inline void add(ByteCount& total, ByteCount value)
{
    total.live += value.live;
    total.capacity += value.capacity;
}

template <typename T>
ByteCount bytes(const thrust::device_vector<T>& values)
{
    return {values.size() * sizeof(T), values.capacity() * sizeof(T)};
}

template <typename T>
ByteCount bytes(const std::vector<T>& values)
{
    return {values.size() * sizeof(T), values.capacity() * sizeof(T)};
}

template <typename T>
void emit_buffer(
    std::ostream& stream,
    std::string_view name,
    const thrust::device_vector<T>& values
)
{
    const ByteCount count = bytes(values);
    stream << ' ' << name << "_live_bytes=" << count.live
           << ' ' << name << "_capacity_bytes=" << count.capacity;
}

template <typename T>
void emit_buffer(std::ostream& stream, std::string_view name, const std::vector<T>& values)
{
    const ByteCount count = bytes(values);
    stream << ' ' << name << "_live_bytes=" << count.live
           << ' ' << name << "_capacity_bytes=" << count.capacity;
}

inline ByteCount cuda_workspace_bytes(const cuda_detail::CudaWorkspace& workspace)
{
    ByteCount total;
#define RI_ADD_DEVICE_FIELD(field) add(total, bytes(workspace.field))
    RI_ADD_DEVICE_FIELD(points);
    RI_ADD_DEVICE_FIELD(accumulated_vertices_global);
    RI_ADD_DEVICE_FIELD(current_vertex_normal_sums);
    RI_ADD_DEVICE_FIELD(accumulated_faces_global);
    RI_ADD_DEVICE_FIELD(current_vertices_local);
    RI_ADD_DEVICE_FIELD(accumulated_depth);
    RI_ADD_DEVICE_FIELD(projected_rows);
    RI_ADD_DEVICE_FIELD(projected_cols);
    RI_ADD_DEVICE_FIELD(projected_ranges);
    RI_ADD_DEVICE_FIELD(pixel_last_index);
    RI_ADD_DEVICE_FIELD(valid);
    RI_ADD_DEVICE_FIELD(clean_valid);
    RI_ADD_DEVICE_FIELD(meshing_valid);
    RI_ADD_DEVICE_FIELD(new_vertex_allowed);
    RI_ADD_DEVICE_FIELD(final_new_allowed);
    RI_ADD_DEVICE_FIELD(ri_thin_input);
    RI_ADD_DEVICE_FIELD(ri_thin_eroded);
    RI_ADD_DEVICE_FIELD(ri_thin_grown);
    RI_ADD_DEVICE_FIELD(ri_thin_removed);
    RI_ADD_DEVICE_FIELD(stitch_connection_mask);
    RI_ADD_DEVICE_FIELD(stitch_connection_old_support);
    RI_ADD_DEVICE_FIELD(stitch_connection_new_support);
    RI_ADD_DEVICE_FIELD(stitch_connection_next_frontier);
    RI_ADD_DEVICE_FIELD(pixel_xyz);
    RI_ADD_DEVICE_FIELD(scan_ranges);
    RI_ADD_DEVICE_FIELD(filled_ranges);
    RI_ADD_DEVICE_FIELD(fill_scratch_ranges);
    RI_ADD_DEVICE_FIELD(fill_support_ranges);
    RI_ADD_DEVICE_FIELD(fill_anchor);
    RI_ADD_DEVICE_FIELD(fill_scratch_anchor);
    RI_ADD_DEVICE_FIELD(compacted_current_vertices);
    RI_ADD_DEVICE_FIELD(filled_pixel_xyz);
    RI_ADD_DEVICE_FIELD(current_vertex_rows);
    RI_ADD_DEVICE_FIELD(current_vertex_cols);
    RI_ADD_DEVICE_FIELD(current_vertex_ranges);
    RI_ADD_DEVICE_FIELD(flags);
    RI_ADD_DEVICE_FIELD(offsets);
    RI_ADD_DEVICE_FIELD(large_face_indices);
    RI_ADD_DEVICE_FIELD(filled_vim);
    RI_ADD_DEVICE_FIELD(macro_quads);
    RI_ADD_DEVICE_FIELD(macro_quad_count);
    RI_ADD_DEVICE_FIELD(macro_candidate_flags);
    RI_ADD_DEVICE_FIELD(macro_selected_size);
    RI_ADD_DEVICE_FIELD(macro_next_size);
    RI_ADD_DEVICE_FIELD(face_records);
    RI_ADD_DEVICE_FIELD(face_record_count);
    RI_ADD_DEVICE_FIELD(face_overflow);
    RI_ADD_DEVICE_FIELD(quad_face_flags);
    RI_ADD_DEVICE_FIELD(faces);
    RI_ADD_DEVICE_FIELD(edge_keys);
    RI_ADD_DEVICE_FIELD(oriented_boundary_edges);
    RI_ADD_DEVICE_FIELD(boundary_edges);
    RI_ADD_DEVICE_FIELD(active_face_ranges);
    RI_ADD_DEVICE_FIELD(active_vertex_ranges);
#undef RI_ADD_DEVICE_FIELD
    return total;
}

inline ByteCount accumulation_bytes(const cuda_detail::CudaWorkspace& workspace)
{
    ByteCount total;
    add(total, bytes(workspace.accumulated_vertices_global));
    add(total, bytes(workspace.accumulated_faces_global));
    return total;
}

inline ByteCount stitch_host_bytes(const StitchWorkspace& workspace)
{
    ByteCount total;
    add(total, bytes(workspace.host_vertices_global));
    add(total, bytes(workspace.host_current_vertex_normal_sums));
    add(total, bytes(workspace.host_vertex_normal_sums_global));
    add(total, bytes(workspace.filtered_old_boundary.edges));
    add(total, bytes(workspace.filtered_old_boundary.old_vertex_ids));
    add(total, bytes(workspace.filtered_old_boundary.new_vertex_ids));
    add(total, bytes(workspace.stitch_global_to_compact));
    add(total, bytes(workspace.pending_boundary_vertex_changes));
    add(total, bytes(workspace.pending_boundary_edges));
    add(total, bytes(workspace.new_boundary_edges));
    add(total, bytes(workspace.candidate_ownership_workspace.owners));
    add(total, bytes(workspace.candidate_ownership_workspace.generations));
    return total;
}

inline ByteCount host_auxiliary_bytes(const CudaPipelineWorkspace& workspace)
{
    ByteCount total;
    add(total, bytes(workspace.cuda.host_boundary_edges));
    add(total, bytes(workspace.prefetch.cuda.host_boundary_edges));
    add(total, bytes(workspace.cuda.host_stitch_connection_mask.pixels));
    add(total, bytes(workspace.cuda.host_stitch_connection_mask.old_support));
    add(total, bytes(workspace.cuda.host_stitch_connection_mask.new_support));
    add(total, bytes(workspace.prefetch.cuda.host_stitch_connection_mask.pixels));
    add(total, bytes(workspace.prefetch.cuda.host_stitch_connection_mask.old_support));
    add(total, bytes(workspace.prefetch.cuda.host_stitch_connection_mask.new_support));
    add(total, bytes(workspace.accumulated_frame_chunks));
    return total;
}

inline std::size_t process_rss_bytes()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) != 0) {
            continue;
        }
        const std::size_t first_digit = line.find_first_of("0123456789");
        if (first_digit == std::string::npos) {
            return 0;
        }
        return std::stoull(line.substr(first_digit)) * 1024ULL;
    }
    return 0;
}

inline std::size_t gpu_used_bytes()
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    return total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
}

}  // namespace memory_detail

inline void record_memory_snapshot(
    const CudaPipelineWorkspace& workspace,
    int frame_index,
    std::string_view stage
)
{
    const memory_detail::ByteCount main_cuda = memory_detail::cuda_workspace_bytes(workspace.cuda);
    const memory_detail::ByteCount prefetch_cuda = memory_detail::cuda_workspace_bytes(workspace.prefetch.cuda);
    const memory_detail::ByteCount accumulation = memory_detail::accumulation_bytes(workspace.cuda);
    const memory_detail::ByteCount stitch_host = memory_detail::stitch_host_bytes(workspace.stitch);
    const memory_detail::ByteCount host_auxiliary = memory_detail::host_auxiliary_bytes(workspace);
    const std::size_t cuda_live = main_cuda.live + prefetch_cuda.live;
    const std::size_t cuda_capacity = main_cuda.capacity + prefetch_cuda.capacity;
    const std::size_t host_live = stitch_host.live + host_auxiliary.live;
    const std::size_t host_capacity = stitch_host.capacity + host_auxiliary.capacity;
    std::cerr << "memory.stage frame=" << frame_index
              << " stage=" << stage
              << " pass_kind=" << ri_meshing::current_memory_pass_kind
              << " run_index=" << ri_meshing::current_memory_run_index
              << " cpu_rss_bytes=" << memory_detail::process_rss_bytes()
              << " host_live_bytes=" << host_live
              << " host_capacity_bytes=" << host_capacity
              << " stitch_host_live_bytes=" << stitch_host.live
              << " stitch_host_capacity_bytes=" << stitch_host.capacity
              << " cuda_live_bytes=" << cuda_live
              << " cuda_capacity_bytes=" << cuda_capacity
              << " main_cuda_live_bytes=" << main_cuda.live
              << " main_cuda_capacity_bytes=" << main_cuda.capacity
              << " prefetch_cuda_live_bytes=" << prefetch_cuda.live
              << " prefetch_cuda_capacity_bytes=" << prefetch_cuda.capacity
              << " accumulated_cuda_live_bytes=" << accumulation.live
              << " accumulated_cuda_capacity_bytes=" << accumulation.capacity
              << " gpu_used_bytes=" << memory_detail::gpu_used_bytes();
#define RI_EMIT_DEVICE_FIELD(prefix, field) \
    memory_detail::emit_buffer(std::cerr, prefix "." #field, workspace.cuda.field)
    RI_EMIT_DEVICE_FIELD("main_cuda", points);
    RI_EMIT_DEVICE_FIELD("main_cuda", accumulated_vertices_global);
    RI_EMIT_DEVICE_FIELD("main_cuda", current_vertex_normal_sums);
    RI_EMIT_DEVICE_FIELD("main_cuda", accumulated_faces_global);
    RI_EMIT_DEVICE_FIELD("main_cuda", current_vertices_local);
    RI_EMIT_DEVICE_FIELD("main_cuda", accumulated_depth);
    RI_EMIT_DEVICE_FIELD("main_cuda", projected_rows);
    RI_EMIT_DEVICE_FIELD("main_cuda", projected_cols);
    RI_EMIT_DEVICE_FIELD("main_cuda", projected_ranges);
    RI_EMIT_DEVICE_FIELD("main_cuda", pixel_last_index);
    RI_EMIT_DEVICE_FIELD("main_cuda", valid);
    RI_EMIT_DEVICE_FIELD("main_cuda", clean_valid);
    RI_EMIT_DEVICE_FIELD("main_cuda", meshing_valid);
    RI_EMIT_DEVICE_FIELD("main_cuda", new_vertex_allowed);
    RI_EMIT_DEVICE_FIELD("main_cuda", final_new_allowed);
    RI_EMIT_DEVICE_FIELD("main_cuda", ri_thin_input);
    RI_EMIT_DEVICE_FIELD("main_cuda", ri_thin_eroded);
    RI_EMIT_DEVICE_FIELD("main_cuda", ri_thin_grown);
    RI_EMIT_DEVICE_FIELD("main_cuda", ri_thin_removed);
    RI_EMIT_DEVICE_FIELD("main_cuda", stitch_connection_mask);
    RI_EMIT_DEVICE_FIELD("main_cuda", stitch_connection_old_support);
    RI_EMIT_DEVICE_FIELD("main_cuda", stitch_connection_new_support);
    RI_EMIT_DEVICE_FIELD("main_cuda", stitch_connection_next_frontier);
    RI_EMIT_DEVICE_FIELD("main_cuda", pixel_xyz);
    RI_EMIT_DEVICE_FIELD("main_cuda", scan_ranges);
    RI_EMIT_DEVICE_FIELD("main_cuda", filled_ranges);
    RI_EMIT_DEVICE_FIELD("main_cuda", fill_scratch_ranges);
    RI_EMIT_DEVICE_FIELD("main_cuda", fill_support_ranges);
    RI_EMIT_DEVICE_FIELD("main_cuda", fill_anchor);
    RI_EMIT_DEVICE_FIELD("main_cuda", fill_scratch_anchor);
    RI_EMIT_DEVICE_FIELD("main_cuda", compacted_current_vertices);
    RI_EMIT_DEVICE_FIELD("main_cuda", filled_pixel_xyz);
    RI_EMIT_DEVICE_FIELD("main_cuda", current_vertex_rows);
    RI_EMIT_DEVICE_FIELD("main_cuda", current_vertex_cols);
    RI_EMIT_DEVICE_FIELD("main_cuda", current_vertex_ranges);
    RI_EMIT_DEVICE_FIELD("main_cuda", flags);
    RI_EMIT_DEVICE_FIELD("main_cuda", offsets);
    RI_EMIT_DEVICE_FIELD("main_cuda", large_face_indices);
    RI_EMIT_DEVICE_FIELD("main_cuda", filled_vim);
    RI_EMIT_DEVICE_FIELD("main_cuda", macro_quads);
    RI_EMIT_DEVICE_FIELD("main_cuda", macro_quad_count);
    RI_EMIT_DEVICE_FIELD("main_cuda", macro_candidate_flags);
    RI_EMIT_DEVICE_FIELD("main_cuda", macro_selected_size);
    RI_EMIT_DEVICE_FIELD("main_cuda", macro_next_size);
    RI_EMIT_DEVICE_FIELD("main_cuda", face_records);
    RI_EMIT_DEVICE_FIELD("main_cuda", face_record_count);
    RI_EMIT_DEVICE_FIELD("main_cuda", face_overflow);
    RI_EMIT_DEVICE_FIELD("main_cuda", quad_face_flags);
    RI_EMIT_DEVICE_FIELD("main_cuda", faces);
    RI_EMIT_DEVICE_FIELD("main_cuda", edge_keys);
    RI_EMIT_DEVICE_FIELD("main_cuda", oriented_boundary_edges);
    RI_EMIT_DEVICE_FIELD("main_cuda", boundary_edges);
    RI_EMIT_DEVICE_FIELD("main_cuda", active_face_ranges);
    RI_EMIT_DEVICE_FIELD("main_cuda", active_vertex_ranges);
#define RI_EMIT_PREFETCH_FIELD(field) \
    memory_detail::emit_buffer(std::cerr, "prefetch_cuda." #field, workspace.prefetch.cuda.field)
    RI_EMIT_PREFETCH_FIELD(points);
    RI_EMIT_PREFETCH_FIELD(accumulated_vertices_global);
    RI_EMIT_PREFETCH_FIELD(current_vertex_normal_sums);
    RI_EMIT_PREFETCH_FIELD(accumulated_faces_global);
    RI_EMIT_PREFETCH_FIELD(current_vertices_local);
    RI_EMIT_PREFETCH_FIELD(accumulated_depth);
    RI_EMIT_PREFETCH_FIELD(projected_rows);
    RI_EMIT_PREFETCH_FIELD(projected_cols);
    RI_EMIT_PREFETCH_FIELD(projected_ranges);
    RI_EMIT_PREFETCH_FIELD(pixel_last_index);
    RI_EMIT_PREFETCH_FIELD(valid);
    RI_EMIT_PREFETCH_FIELD(clean_valid);
    RI_EMIT_PREFETCH_FIELD(meshing_valid);
    RI_EMIT_PREFETCH_FIELD(new_vertex_allowed);
    RI_EMIT_PREFETCH_FIELD(final_new_allowed);
    RI_EMIT_PREFETCH_FIELD(ri_thin_input);
    RI_EMIT_PREFETCH_FIELD(ri_thin_eroded);
    RI_EMIT_PREFETCH_FIELD(ri_thin_grown);
    RI_EMIT_PREFETCH_FIELD(ri_thin_removed);
    RI_EMIT_PREFETCH_FIELD(stitch_connection_mask);
    RI_EMIT_PREFETCH_FIELD(stitch_connection_old_support);
    RI_EMIT_PREFETCH_FIELD(stitch_connection_new_support);
    RI_EMIT_PREFETCH_FIELD(stitch_connection_next_frontier);
    RI_EMIT_PREFETCH_FIELD(pixel_xyz);
    RI_EMIT_PREFETCH_FIELD(scan_ranges);
    RI_EMIT_PREFETCH_FIELD(filled_ranges);
    RI_EMIT_PREFETCH_FIELD(fill_scratch_ranges);
    RI_EMIT_PREFETCH_FIELD(fill_support_ranges);
    RI_EMIT_PREFETCH_FIELD(fill_anchor);
    RI_EMIT_PREFETCH_FIELD(fill_scratch_anchor);
    RI_EMIT_PREFETCH_FIELD(compacted_current_vertices);
    RI_EMIT_PREFETCH_FIELD(filled_pixel_xyz);
    RI_EMIT_PREFETCH_FIELD(current_vertex_rows);
    RI_EMIT_PREFETCH_FIELD(current_vertex_cols);
    RI_EMIT_PREFETCH_FIELD(current_vertex_ranges);
    RI_EMIT_PREFETCH_FIELD(flags);
    RI_EMIT_PREFETCH_FIELD(offsets);
    RI_EMIT_PREFETCH_FIELD(large_face_indices);
    RI_EMIT_PREFETCH_FIELD(filled_vim);
    RI_EMIT_PREFETCH_FIELD(macro_quads);
    RI_EMIT_PREFETCH_FIELD(macro_quad_count);
    RI_EMIT_PREFETCH_FIELD(macro_candidate_flags);
    RI_EMIT_PREFETCH_FIELD(macro_selected_size);
    RI_EMIT_PREFETCH_FIELD(macro_next_size);
    RI_EMIT_PREFETCH_FIELD(face_records);
    RI_EMIT_PREFETCH_FIELD(face_record_count);
    RI_EMIT_PREFETCH_FIELD(face_overflow);
    RI_EMIT_PREFETCH_FIELD(quad_face_flags);
    RI_EMIT_PREFETCH_FIELD(faces);
    RI_EMIT_PREFETCH_FIELD(edge_keys);
    RI_EMIT_PREFETCH_FIELD(oriented_boundary_edges);
    RI_EMIT_PREFETCH_FIELD(boundary_edges);
    RI_EMIT_PREFETCH_FIELD(active_face_ranges);
    RI_EMIT_PREFETCH_FIELD(active_vertex_ranges);
#undef RI_EMIT_PREFETCH_FIELD
#undef RI_EMIT_DEVICE_FIELD
    memory_detail::emit_buffer(std::cerr, "host.stitch.host_vertices_global", workspace.stitch.host_vertices_global);
    memory_detail::emit_buffer(std::cerr, "host.stitch.host_current_vertex_normal_sums", workspace.stitch.host_current_vertex_normal_sums);
    memory_detail::emit_buffer(std::cerr, "host.stitch.host_vertex_normal_sums_global", workspace.stitch.host_vertex_normal_sums_global);
    memory_detail::emit_buffer(std::cerr, "host.stitch.filtered_old_boundary_edges", workspace.stitch.filtered_old_boundary.edges);
    memory_detail::emit_buffer(std::cerr, "host.stitch.filtered_old_boundary_old_vertex_ids", workspace.stitch.filtered_old_boundary.old_vertex_ids);
    memory_detail::emit_buffer(std::cerr, "host.stitch.filtered_old_boundary_new_vertex_ids", workspace.stitch.filtered_old_boundary.new_vertex_ids);
    memory_detail::emit_buffer(std::cerr, "host.stitch.stitch_global_to_compact", workspace.stitch.stitch_global_to_compact);
    memory_detail::emit_buffer(std::cerr, "host.stitch.pending_boundary_vertex_changes", workspace.stitch.pending_boundary_vertex_changes);
    memory_detail::emit_buffer(std::cerr, "host.stitch.pending_boundary_edges", workspace.stitch.pending_boundary_edges);
    memory_detail::emit_buffer(std::cerr, "host.stitch.new_boundary_edges", workspace.stitch.new_boundary_edges);
    memory_detail::emit_buffer(std::cerr, "host.stitch.candidate_ownership_owners", workspace.stitch.candidate_ownership_workspace.owners);
    memory_detail::emit_buffer(std::cerr, "host.stitch.candidate_ownership_generations", workspace.stitch.candidate_ownership_workspace.generations);
    memory_detail::emit_buffer(std::cerr, "host.main.host_boundary_edges", workspace.cuda.host_boundary_edges);
    memory_detail::emit_buffer(std::cerr, "host.prefetch.host_boundary_edges", workspace.prefetch.cuda.host_boundary_edges);
    memory_detail::emit_buffer(std::cerr, "host.main.connection_mask_pixels", workspace.cuda.host_stitch_connection_mask.pixels);
    memory_detail::emit_buffer(std::cerr, "host.main.connection_mask_old_support", workspace.cuda.host_stitch_connection_mask.old_support);
    memory_detail::emit_buffer(std::cerr, "host.main.connection_mask_new_support", workspace.cuda.host_stitch_connection_mask.new_support);
    memory_detail::emit_buffer(std::cerr, "host.prefetch.connection_mask_pixels", workspace.prefetch.cuda.host_stitch_connection_mask.pixels);
    memory_detail::emit_buffer(std::cerr, "host.prefetch.connection_mask_old_support", workspace.prefetch.cuda.host_stitch_connection_mask.old_support);
    memory_detail::emit_buffer(std::cerr, "host.prefetch.connection_mask_new_support", workspace.prefetch.cuda.host_stitch_connection_mask.new_support);
    memory_detail::emit_buffer(std::cerr, "host.accumulated_frame_chunks", workspace.accumulated_frame_chunks);
    std::cerr << '\n';
}

}  // namespace ri_meshing::cuda_pipeline

#else

namespace ri_meshing::cuda_pipeline {

inline void record_memory_snapshot(
    const CudaPipelineWorkspace&,
    int,
    std::string_view
)
{
}

}  // namespace ri_meshing::cuda_pipeline

#endif
