#include "stitch_transaction.hpp"

#include "frame_overlap_scheduler.hpp"
#include "ri_meshing/geometry.hpp"
#include "ri_meshing/hole_filling.hpp"
#include "stitch_compaction.hpp"
#include "stitch_debug.hpp"
#include "stitch_geometry.hpp"
#include "stitch_gpu_helpers.hpp"
#include "stitch_proxy_cpu.hpp"
#include "stitch_spatial.hpp"
#include "../stitch_connection.hpp"
#include "device_vector_memory.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace ri_meshing::cuda_pipeline {
namespace {

void update_host_boundary(
    BoundaryGraph& host_boundary,
    std::vector<int>& pending_changes,
    const std::vector<std::array<int, 2>>& new_edges,
    const StitchCoreResult& stitch_result,
    std::vector<std::array<int, 2>>& stitch_edges_out
)
{
    stitch_edges_out.clear();
    std::vector<std::array<int, 2>> combined_edges = new_edges;
    if (!stitch_result.faces.empty()) {
        stitch_edges_out = batch_boundary_edges_sorted(stitch_result.faces);
        combined_edges.insert(
            combined_edges.end(), stitch_edges_out.begin(), stitch_edges_out.end()
        );
    }
    const std::vector<int> changes = host_boundary.toggle_boundary_edges(combined_edges);
    pending_changes.insert(pending_changes.end(), changes.begin(), changes.end());
}

void append_stitch_faces(
    cuda_detail::CudaWorkspace& workspace,
    cuda_detail::CudaAccumulationCounts& counts,
    const std::vector<Triangle>& stitch_faces,
    PipelineTiming* timing
)
{
    cuda_detail::timed_cuda_stage_void(timing, "stitch.append_faces", [&] {
        cuda_detail::resize_with_headroom(
            workspace.accumulated_faces_global,
            static_cast<std::size_t>(counts.face_count + static_cast<int>(stitch_faces.size()))
        );
        cuda_detail::copy_host_to_device_sync(
            thrust::raw_pointer_cast(workspace.accumulated_faces_global.data()) + counts.face_count,
            stitch_faces.data(),
            stitch_faces.size()
        );
        counts.face_count += static_cast<int>(stitch_faces.size());
    });
}

}  // namespace

StitchTransaction::StitchTransaction(
    StitchTransactionResources resources,
    const StitchFrameInput& input,
    cuda_detail::CudaAccumulationCounts& counts,
    PipelineTiming* timing
)
    : resources_(resources)
    , input_(input)
    , counts_(counts)
    , timing_(timing)
    , new_boundary_edges_(resources.workspace.stitch.new_boundary_edges)
{
}

StitchAppendResult StitchTransaction::execute()
{
    if (!should_run()) {
        return {};
    }
    validate();
    extract_boundary();
    prepare_filter_compact();
    solve();
    log_core_debug();
    update_boundary();
    if (resources_.enable_hole_filling) {
        fill_local_holes();
    }
    finish_proxy_metrics();
    return append_result();
}

bool StitchTransaction::should_run() const
{
    return input_.new_face_count > 0;
}

bool StitchTransaction::proxy_metrics_enabled() const
{
    return resources_.proxy_controller.collects_metrics();
}

bool StitchTransaction::proxy_blocker_enabled() const
{
    return resources_.proxy_controller.uses_blocker();
}

const Mat4f* StitchTransaction::next_pose() const
{
    return input_.next_pose ? &*input_.next_pose : nullptr;
}

const StitchDebugDump* StitchTransaction::debug_dump() const
{
    return resources_.debug ? &*resources_.debug : nullptr;
}

void StitchTransaction::validate() const
{
#if !defined(RI_MESHING_ENABLE_DEBUG_EXPORT)
    if (debug_dump() != nullptr) {
        throw std::runtime_error("Stitch debug export was not compiled in this build");
    }
#endif
}

void StitchTransaction::extract_boundary()
{
    const std::string edge_count = "edges=" + std::to_string(new_boundary_edges_.size());
    log("stitch.new_boundary_ready", edge_count);
    download_debug_faces();
}

void StitchTransaction::download_debug_faces()
{
#if defined(RI_MESHING_ENABLE_DEBUG_EXPORT)
    if (debug_dump() == nullptr || input_.old_face_count <= 0) {
        return;
    }
    cuda_detail::timed_cuda_stage_void(timing_, "debug.download_new_faces", [&] {
        new_faces_.resize(static_cast<std::size_t>(input_.new_face_count));
        cuda_detail::copy_device_to_host_sync(
            new_faces_.data(),
            thrust::raw_pointer_cast(resources_.workspace.cuda.accumulated_faces_global.data())
                + input_.old_face_count,
            static_cast<std::size_t>(input_.new_face_count)
        );
    });
#endif
}

void StitchTransaction::prepare_filter_compact()
{
    log("stitch.compact_input_start");
    cuda_detail::timed_cuda_stage_void(timing_, "stitch.compact_input", [&] {
        prepare_compact_input();
    });
    log(
        "stitch.compact_input_done",
        "compact_vertices=" + std::to_string(compact_input_.compact_to_global.size())
    );
}

void StitchTransaction::prepare_compact_input()
{
    if (input_.old_vertex_count <= 0 || input_.old_face_count <= 0) {
        return;
    }
    Stopwatch compact_sub_stopwatch;
    const Mat4f global_to_local = inverse_rigid_cpu(input_.pose);
    if (timing_ != nullptr) {
        timing_->add_stage("stitch.compact.inverse_pose", compact_sub_stopwatch.elapsed_milliseconds());
    }

    compact_sub_stopwatch = Stopwatch{};
    const FilteredOldBoundaryEdges& filtered = filter_old_boundary();
    prepare_proxy_metrics();
    if (timing_ != nullptr) {
        timing_->add_stage("stitch.compact.host_boundary_cache", compact_sub_stopwatch.elapsed_milliseconds());
    }

    compact_sub_stopwatch = Stopwatch{};
    compact_filtered_input(filtered, global_to_local);
    if (timing_ != nullptr) {
        timing_->add_stage("stitch.compact.make_input", compact_sub_stopwatch.elapsed_milliseconds());
    }
    log_compact_debug(filtered);
}

const FilteredOldBoundaryEdges& StitchTransaction::filter_old_boundary()
{
    apply_pending_boundary_changes();
    StitchWorkspace& stitch = resources_.workspace.stitch;
    filter_old_boundary_edges_near_new(
        SpatialBoundaryFilterRequest{
            stitch.host_vertices_global,
            stitch.host_boundary.boundary_edge_blocks(),
            stitch.host_boundary.boundary_edge_count(),
            stitch.host_boundary.boundary_vertex_ids(),
            new_boundary_edges_,
            input_.old_vertex_count,
            resources_.config,
        },
        stitch.stitch_spatial_boundary_index,
        stitch.filtered_old_boundary,
        timing_
    );
    return stitch.filtered_old_boundary;
}

void StitchTransaction::apply_pending_boundary_changes()
{
    StitchWorkspace& stitch = resources_.workspace.stitch;
    if (stitch.pending_boundary_vertex_changes.empty()
        && stitch.stitch_spatial_boundary_index.initialized) {
        return;
    }
    Stopwatch pending_stopwatch;
    apply_active_boundary_changes(
        stitch.stitch_spatial_boundary_index,
        stitch.host_vertices_global,
        stitch.host_boundary.boundary_vertex_ids(),
        stitch.pending_boundary_vertex_changes,
        stitch_spatial_filter_radius(resources_.config)
    );
    stitch.pending_boundary_vertex_changes.clear();
    if (timing_ != nullptr) {
        timing_->add_stage(
            "stitch.compact.filter_old_boundary.apply_pending_changes",
            pending_stopwatch.elapsed_milliseconds()
        );
    }
}

void StitchTransaction::prepare_proxy_metrics()
{
    if (!proxy_metrics_enabled() || next_pose() == nullptr) {
        return;
    }
    const BoundaryEdges empty_edges;
    proxy_scratch_ = build_stitch_proxy_scratch(
        StitchProxyBuildRequest{
            resources_.workspace.stitch.host_vertices_global,
            empty_edges,
            empty_edges,
            empty_edges,
            *next_pose(),
            resources_.config,
            true,
        }
    );
}

void StitchTransaction::compact_filtered_input(
    const FilteredOldBoundaryEdges& filtered,
    const Mat4f& global_to_local
)
{
    StitchWorkspace& stitch = resources_.workspace.stitch;
    compact_stitch_input(
        StitchCompactionRequest{
            stitch.host_vertices_global,
            stitch.host_vertex_normal_sums_global,
            BoundarySelection{filtered.edges, filtered.old_vertex_ids},
            BoundarySelection{new_boundary_edges_, filtered.new_vertex_ids},
            input_.old_vertex_count,
            global_to_local,
        },
        compact_input_,
        stitch.stitch_global_to_compact,
        timing_
    );
}

void StitchTransaction::log_compact_debug(const FilteredOldBoundaryEdges& filtered) const
{
    if (std::getenv("RI_STITCH_DEBUG") == nullptr) {
        return;
    }
    const BoundaryGraph& boundary = resources_.workspace.stitch.host_boundary;
    std::cerr << "stitch.incremental_input debug:"
              << " old_boundary_edges=" << boundary.boundary_edges().size()
              << " filtered_old_boundary_edges=" << filtered.edges.size()
              << " new_faces=" << new_faces_.size()
              << " new_boundary_edges=" << new_boundary_edges_.size()
              << " compact_vertices=" << compact_input_.vertices_local.size()
              << " compact_old_vertices=" << compact_input_.old_vertex_count << '\n';
}

void StitchTransaction::solve()
{
    log("stitch.core_start");
    cuda_detail::timed_cuda_stage_void(timing_, "stitch.core_v0", [&] {
        run_stitch_core();
    });
    log("stitch.core_done", "faces=" + std::to_string(result_.faces.size()));
}

void StitchTransaction::run_stitch_core()
{
    if (compact_input_.old_vertex_count <= 0 || compact_input_.new_boundary_edges.empty()) {
        return;
    }
    StitchWorkspace& stitch = resources_.workspace.stitch;
#if defined(RI_MESHING_ENABLE_DEBUG_EXPORT)
    StitchCoreDebug* core_debug = debug_dump() != nullptr ? &debug_ : nullptr;
#else
    StitchCoreDebug* core_debug = nullptr;
#endif
    result_ = stitch_core(
        StitchCoreInput{
            compact_input_.vertices_local,
            compact_input_.vertex_normals_local,
            compact_input_.old_boundary_edges,
            compact_input_.new_boundary_edges,
            timing_,
            compact_input_.old_vertex_count,
            core_debug,
            polyline_proxy_callback(),
            input_.connection_mask,
            &stitch.stitch_path_workspace,
            &stitch.candidate_ownership_workspace,
        },
        resources_.config
    );
#if defined(RI_MESHING_ENABLE_DEBUG_EXPORT)
    if (debug_dump() != nullptr) {
        remap_stitch_core_debug(debug_, compact_input_.compact_to_global);
    }
#endif
    remap_faces_to_global(result_.faces, compact_input_.compact_to_global);
}

StitchTransaction::PolylineProxyCallback StitchTransaction::polyline_proxy_callback()
{
    const bool enabled = resources_.proxy_controller.paths_matched(next_pose() != nullptr);
    if (!enabled) {
        return {};
    }
    return [this](const BoundaryEdges& compact_edges) {
        publish_polyline_proxy(compact_edges);
    };
}

void StitchTransaction::publish_polyline_proxy(const BoundaryEdges& compact_edges)
{
    const Stopwatch host_prepare_stopwatch;
    StitchProxyEdgeSet global_edges;
    global_edges.bridge_edges.reserve(compact_edges.size());
    for (const std::array<int, 2> edge : compact_edges) {
        const std::size_t first = static_cast<std::size_t>(edge[0]);
        const std::size_t second = static_cast<std::size_t>(edge[1]);
        if (edge[0] < 0 || edge[1] < 0
            || first >= compact_input_.compact_to_global.size()
            || second >= compact_input_.compact_to_global.size()) {
            continue;
        }
        global_edges.bridge_edges.push_back({
            compact_input_.compact_to_global[first],
            compact_input_.compact_to_global[second],
        });
    }
    if (resources_.proxy_controller.measures_kernel_timing()) {
        std::cerr << "stitch.proxy.kernel_timing"
                  << " mode=polyline"
                  << " frame=" << input_.frame_index
                  << " phase=prepare_polyline_set"
                  << " items=" << global_edges.bridge_edges.size()
                  << " host_ms=" << host_prepare_stopwatch.elapsed_milliseconds()
                  << " gpu_ms=0\n";
    }
    proxy_edges_ = std::move(global_edges);
    if (proxy_metrics_enabled()) {
        const BoundaryEdges empty_edges;
        const Stopwatch proxy_stopwatch;
        proxy_scratch_ = build_stitch_proxy_scratch(
            StitchProxyBuildRequest{
                resources_.workspace.stitch.host_vertices_global,
                empty_edges,
                empty_edges,
                proxy_edges_.bridge_edges,
                *next_pose(),
                resources_.config,
                true,
            }
        );
        proxy_scratch_.metrics.build_candidates_milliseconds = proxy_stopwatch.elapsed_milliseconds();
        proxy_scratch_.metrics.milliseconds = proxy_scratch_.metrics.build_candidates_milliseconds;
        if (timing_ != nullptr) {
            timing_->add_stage(
                "stitch.proxy.build_polyline",
                proxy_scratch_.metrics.build_candidates_milliseconds
            );
        }
    }
    proxy_overlap_started_ = resources_.scheduler.start_proxy_prefetch(
        ProxyPrefetchRequest{
            input_.frame_position,
            counts_,
            proxy_edges_,
            input_.old_face_count + input_.new_face_count,
        }
    );
    log(
        proxy_overlap_started_ ? "stitch.polyline_proxy_started" : "stitch.polyline_proxy_deferred",
        "bridge_edges=" + std::to_string(proxy_edges_.bridge_edges.size())
    );
}

void StitchTransaction::finish_proxy_metrics()
{
    if (!proxy_metrics_enabled() || !proxy_scratch_.metrics.evaluated || next_pose() == nullptr) {
        return;
    }
    cuda_pipeline::finish_stitch_proxy_metrics(
        StitchProxyMetricsRequest{
            proxy_scratch_,
            resources_.workspace.stitch.host_vertices_global,
            StitchProxyOracleFaces{result_.faces, hole_faces_},
            *next_pose(),
            resources_.config,
            resources_.proxy_controller.metrics_output_dir().empty()
                ? nullptr : &resources_.proxy_controller.metrics_output_dir(),
            input_.frame_index,
        }
    );
    const double proxy_ms = proxy_scratch_.metrics.raster_oracle_milliseconds
        + proxy_scratch_.metrics.compare_masks_milliseconds;
    proxy_scratch_.metrics.milliseconds += proxy_ms;
    if (timing_ != nullptr) {
        timing_->add_stage("stitch.proxy.raster_oracle", proxy_scratch_.metrics.raster_oracle_milliseconds);
        timing_->add_stage("stitch.proxy.compare_masks", proxy_scratch_.metrics.compare_masks_milliseconds);
    }
}

void StitchTransaction::log_core_debug() const
{
    if (std::getenv("RI_STITCH_DEBUG") == nullptr) {
        return;
    }
    std::cerr << "stitch.core_v0 debug:"
              << " host_edges_before_add=" << resources_.workspace.stitch.host_boundary.boundary_edge_count()
              << " old_boundary_edges_used=" << result_.old_boundary_edges_used
              << " old_boundary_vertices=" << result_.old_boundary_vertices
              << " new_boundary_vertices=" << result_.new_boundary_vertices
              << " candidate_faces=" << result_.candidate_faces
              << " selected_faces=" << result_.faces.size() << '\n';
}

void StitchTransaction::update_boundary()
{
    log("stitch.update_host_boundary_start");
    cuda_detail::timed_cuda_stage_void(timing_, "stitch.update_host_boundary_combined", [&] {
        update_host_boundary(
            resources_.workspace.stitch.host_boundary,
            resources_.workspace.stitch.pending_boundary_vertex_changes,
            new_boundary_edges_,
            result_,
            stitch_boundary_edges_
        );
    });
    log("stitch.update_host_boundary_done");
    log("stitch.boundary_combined_updated");
    log_boundary_debug();
}

void StitchTransaction::fill_local_holes()
{
    hole_faces_.clear();
    hole_candidate_loops_ = 0;
    hole_accepted_loops_ = 0;
    if (input_.connection_mask == nullptr) {
        return;
    }
    const Mat4f global_to_local = inverse_rigid_cpu(input_.pose);
    const Stopwatch path_stopwatch;
    BoundaryPathExtractionStats path_stats;
    const std::vector<BoundaryPath>& oriented_paths =
        resources_.workspace.stitch.host_boundary.oriented_boundary_paths_for_holes(
            stitch_boundary_edges_,
            static_cast<std::size_t>(resources_.config.max_hole_edges),
            timing_ != nullptr ? &path_stats : nullptr
        );
    if (timing_ != nullptr) {
        timing_->add_stage(
            "stitch.hole_fill.extract_oriented_paths",
            path_stopwatch.elapsed_milliseconds()
        );
        timing_->add_stage("stitch.hole_fill.edge_records", path_stats.edge_records_milliseconds);
        timing_->add_stage("stitch.hole_fill.adjacency", path_stats.adjacency_milliseconds);
        timing_->add_stage("stitch.hole_fill.orientation", path_stats.orientation_milliseconds);
        timing_->add_stage("stitch.hole_fill.path_sort", path_stats.final_sort_milliseconds);
        timing_->add_stage("stitch.hole_fill.vertex_catalog", path_stats.vertex_catalog_milliseconds);
        timing_->add_stage("stitch.hole_fill.vertex_lookup", path_stats.vertex_lookup_milliseconds);
        timing_->add_stage("stitch.hole_fill.csr_index", path_stats.csr_index_milliseconds);
        timing_->add_stage("stitch.hole_fill.csr_offsets", path_stats.csr_offsets_milliseconds);
        timing_->add_stage("stitch.hole_fill.csr_fill", path_stats.csr_fill_milliseconds);
        timing_->add_stage("stitch.hole_fill.csr_sort", path_stats.csr_sort_milliseconds);
        timing_->add_stage("stitch.hole_fill.csr_visited", path_stats.csr_visited_milliseconds);
        timing_->add_stage(
            "stitch.hole_fill.dirty_collection", path_stats.dirty_collection_milliseconds
        );
        timing_->add_stage(
            "stitch.hole_fill.cache_publish", path_stats.cache_publish_milliseconds
        );
        log(
            "stitch.hole_fill.path_scope",
            "seed_edges=" + std::to_string(path_stats.input_edges)
                + " unique_seed_edges=" + std::to_string(path_stats.unique_edges)
                + " bounded_walk_vertices=" + std::to_string(path_stats.seeded_walk_vertices)
                + " bounded_walk_steps=" + std::to_string(path_stats.seeded_walk_steps)
                + " candidate_loops=" + std::to_string(path_stats.oriented_paths)
        );
    }
    const HoleFillResult result = fill_local_seam_holes(
        HoleFillRequest{
            resources_.workspace.stitch.host_vertices_global,
            oriented_paths,
            input_.connection_mask,
            resources_.config,
            input_.old_vertex_count,
            global_to_local,
            timing_,
            resources_.hole_filling_debug.has_value(),
        }
    );
    if (resources_.hole_filling_debug.has_value()
        && (resources_.hole_filling_debug->max_frames <= 0
            || resources_.dumped_hole_frames < resources_.hole_filling_debug->max_frames)) {
        dump_hole_fill_metrics(*resources_.hole_filling_debug, input_.frame_index, result);
        ++resources_.dumped_hole_frames;
    }
    const Stopwatch append_stopwatch;
    append_hole_vertices(result.vertices_global);
    if (timing_ != nullptr) {
        timing_->add_stage(
            "stitch.hole_fill.append_vertices",
            append_stopwatch.elapsed_milliseconds()
        );
    }
    hole_faces_ = result.faces;
    hole_candidate_loops_ = result.candidate_loops;
    hole_accepted_loops_ = result.accepted_loops;
    const Stopwatch boundary_stopwatch;
    if (!hole_faces_.empty()) {
        update_boundary_from_faces(hole_faces_);
    }
    if (timing_ != nullptr) {
        timing_->add_stage(
            "stitch.hole_fill.update_boundary",
            boundary_stopwatch.elapsed_milliseconds()
        );
    }
    log(
        "stitch.local_hole_fill_done",
        "candidate_loops=" + std::to_string(hole_candidate_loops_)
            + " accepted_loops=" + std::to_string(hole_accepted_loops_)
            + " faces=" + std::to_string(hole_faces_.size())
            + " support_loops=" + std::to_string(result.support_loops)
            + " center_attempts=" + std::to_string(result.center_attempts)
            + " center_accepted=" + std::to_string(result.center_accepted)
            + " fallback_attempts=" + std::to_string(result.fallback_attempts)
            + " fallback_accepted=" + std::to_string(result.fallback_accepted)
    );
}

void StitchTransaction::append_hole_vertices(const std::vector<Vec3f>& vertices)
{
    if (vertices.empty()) {
        return;
    }
    const int first_vertex = counts_.vertex_count;
    if (resources_.workspace.stitch.host_vertices_global.size()
        != static_cast<std::size_t>(first_vertex)) {
        throw std::runtime_error("host/device vertex counts diverged before local hole fill");
    }
    resources_.workspace.stitch.host_vertices_global.insert(
        resources_.workspace.stitch.host_vertices_global.end(), vertices.begin(), vertices.end()
    );
    const int new_vertex_count = static_cast<int>(vertices.size());
    cuda_detail::resize_with_headroom(
        resources_.workspace.cuda.accumulated_vertices_global,
        static_cast<std::size_t>(first_vertex + new_vertex_count)
    );
    cuda_detail::copy_host_to_device_sync(
        thrust::raw_pointer_cast(resources_.workspace.cuda.accumulated_vertices_global.data()) + first_vertex,
        vertices.data(),
        vertices.size()
    );
    counts_.vertex_count += new_vertex_count;
}

void StitchTransaction::update_boundary_from_faces(const std::vector<Triangle>& faces)
{
    if (faces.empty()) {
        return;
    }
    const std::vector<std::array<int, 2>> edges = batch_boundary_edges_sorted(faces);
    auto& pending_edges = resources_.workspace.stitch.pending_boundary_edges;
    pending_edges.insert(pending_edges.end(), edges.begin(), edges.end());
}

void StitchTransaction::log_boundary_debug() const
{
    if (std::getenv("RI_STITCH_DEBUG") == nullptr) {
        return;
    }
    const BoundaryGraph& boundary = resources_.workspace.stitch.host_boundary;
    const auto histogram = boundary.incidence_histogram();
    std::cerr << "stitch.host_boundary debug:"
              << " added_new_faces=" << new_faces_.size()
              << " host_edges_after_new=" << boundary.boundary_edge_count()
              << " host_boundary_edges=" << boundary.boundary_edge_count()
              << " inc1=" << histogram[0]
              << " inc2=" << histogram[1]
              << " inc3=" << histogram[2]
              << " inc4plus=" << histogram[3] << '\n';
}

StitchAppendResult StitchTransaction::append_result()
{
    if (result_.faces.empty() && hole_faces_.empty()) {
        wait_for_proxy();
        // A requested debug bundle must retain a snapshot even when this
        // frame contributes no stitching or hole-fill faces.  The snapshot
        // still contains the accumulated mesh and boundary state, and is
        // required by the interactive viewer for sparse/no-stitch runs.
        dump_debug_snapshot();
        StitchAppendResult output = make_result();
        return output;
    }
    dump_debug_snapshot();
    wait_for_proxy();
    log("stitch.prefix_wait_done");
    append_faces();
    StitchAppendResult output = make_result();
    return output;
}

void StitchTransaction::dump_debug_snapshot()
{
#if defined(RI_MESHING_ENABLE_DEBUG_EXPORT)
    const StitchDebugDump* dump = debug_dump();
    if (dump == nullptr || (dump->max_frames > 0 && resources_.dumped_frames >= dump->max_frames)) {
        return;
    }
    cuda_detail::timed_cuda_stage_void(timing_, "debug.dump_stitch_snapshot", [&] {
        thrust::host_vector<Triangle> faces_before(static_cast<std::size_t>(counts_.face_count));
        cuda_detail::copy_device_to_host_sync(
            faces_before.data(),
            thrust::raw_pointer_cast(resources_.workspace.cuda.accumulated_faces_global.data()),
            static_cast<std::size_t>(counts_.face_count)
        );
        dump_stitch_debug_snapshot(
            *dump,
            StitchFrameDebugSnapshot{
                input_.frame_index,
                dump->label,
                resources_.workspace.stitch.host_vertices_global,
                std::vector<Triangle>(faces_before.begin(), faces_before.end()),
                new_faces_,
                result_.faces,
                hole_faces_,
                std::move(debug_),
                static_cast<std::size_t>(input_.old_face_count),
                static_cast<std::size_t>(input_.new_face_count),
            }
        );
    });
    ++resources_.dumped_frames;
#endif
}

void StitchTransaction::wait_for_proxy()
{
    log("stitch.proxy_wait_start");
    resources_.scheduler.wait_for_proxy_prefix();
    log("stitch.proxy_wait_done");
}

void StitchTransaction::append_faces()
{
    log_cuda("stitch.append_faces_start");
    append_stitch_faces(resources_.workspace.cuda, counts_, result_.faces, timing_);
    accumulate_host_face_normal_sums(
        resources_.workspace.stitch.host_vertices_global,
        result_.faces,
        resources_.workspace.stitch.host_vertex_normal_sums_global,
        timing_
    );
    append_stitch_faces(resources_.workspace.cuda, counts_, hole_faces_, timing_);
    accumulate_host_face_normal_sums(
        resources_.workspace.stitch.host_vertices_global,
        hole_faces_,
        resources_.workspace.stitch.host_vertex_normal_sums_global,
        timing_
    );
    log_cuda(
        "stitch.append_faces_done",
        "stitch_faces=" + std::to_string(result_.faces.size())
            + " hole_faces=" + std::to_string(hole_faces_.size())
    );
}

StitchAppendResult StitchTransaction::make_result()
{
    return StitchAppendResult{
        result_.faces.size(),
        hole_faces_.size(),
        proxy_scratch_.metrics,
        proxy_blocker_enabled() && !proxy_overlap_started_
            ? std::move(proxy_edges_) : StitchProxyEdgeSet{},
        input_.old_face_count + input_.new_face_count,
    };
}

void StitchTransaction::log(const char* event, const std::string& detail) const
{
    resources_.timeline.log(input_.frame_index, event, detail);
}

void StitchTransaction::log_cuda(const char* event, const std::string& detail) const
{
    resources_.timeline.log_cuda(input_.frame_index, event, 0, detail);
}

}  // namespace ri_meshing::cuda_pipeline
