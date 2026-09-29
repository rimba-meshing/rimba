#include "multi_frame_pipeline_state.hpp"

#include "cuda_utils.hpp"
#include "pipeline_debug.hpp"
#include "overlap_debug.hpp"
#include "stitch_connection_mask.hpp"
#include "stitch_append.hpp"
#include "stitch_geometry.hpp"
#include "stitch_geometry_cuda.cuh"
#include "stitch_gpu_helpers.hpp"
#include "memory_instrumentation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ri_meshing::cuda_pipeline {
namespace {

cuda_detail::StageTimelineCallback frame_timeline(
    OverlapTimeline& timeline,
    bool enabled,
    int frame_index
)
{
    if (!enabled) {
        return {};
    }
    return [&timeline, frame_index](const char* event) {
        timeline.log_cuda(frame_index, event);
    };
}

}  // namespace

namespace {

cuda_detail::AccumulatedFrameChunk make_accumulated_frame_chunk(
    const std::vector<Vec3f>& vertices,
    int first_vertex,
    int vertex_count,
    int first_face,
    int face_count
)
{
    cuda_detail::AccumulatedFrameChunk chunk{
        first_vertex, vertex_count, first_face, face_count, Vec3f{}, 0.0F
    };
    if (vertex_count <= 0) {
        return chunk;
    }
    const std::size_t first = static_cast<std::size_t>(first_vertex);
    const std::size_t end = std::min(
        vertices.size(), first + static_cast<std::size_t>(vertex_count)
    );
    if (first >= end) {
        return chunk;
    }
    Vec3f minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    Vec3f maximum{
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
    };
    for (std::size_t index = first; index < end; ++index) {
        const Vec3f point = vertices[index];
        minimum.x = std::min(minimum.x, point.x);
        minimum.y = std::min(minimum.y, point.y);
        minimum.z = std::min(minimum.z, point.z);
        maximum.x = std::max(maximum.x, point.x);
        maximum.y = std::max(maximum.y, point.y);
        maximum.z = std::max(maximum.z, point.z);
    }
    chunk.center_global = Vec3f{
        (minimum.x + maximum.x) * 0.5F,
        (minimum.y + maximum.y) * 0.5F,
        (minimum.z + maximum.z) * 0.5F,
    };
    float radius_squared = 0.0F;
    for (std::size_t index = first; index < end; ++index) {
        const Vec3f delta{
            vertices[index].x - chunk.center_global.x,
            vertices[index].y - chunk.center_global.y,
            vertices[index].z - chunk.center_global.z,
        };
        radius_squared = std::max(
            radius_squared,
            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
        );
    }
    chunk.radius = std::sqrt(radius_squared);
    return chunk;
}

}  // namespace

FrameExecution::FrameExecution(CudaMultiFramePipeline& pipeline, std::size_t frame_position)
    : pipeline_(pipeline)
    , frame_position_(frame_position)
    , input_(pipeline.frames_[frame_position])
    , stage_timing_(detailed_timing_or_null(pipeline.options_.timing, timing_))
{
}

FrameProcessingStats FrameExecution::run()
{
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "frame_begin");
    apply_pending_boundary_edges();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_pending_boundary");
    acquire_schedule();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_schedule");
    prepare_scan_prefix();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_scan_prefix");
    prepare_accumulated_depth();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_accumulated_depth");
    expose_accumulated_depth();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_depth_exposed");
    build_vertices();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_vertices");
    dump_ri_thin_filter();
    dump_filled_ri();
    build_stitch_connection_mask();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_connection_mask");
    dump_overlap_debug();
    generate_dense_mesh();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_dense_mesh");
    append_dense_mesh();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_dense_append");
    pipeline_.scheduler_.start_pre_scan(frame_position_);
    extract_new_boundary();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_boundary");
    materialize_stitch_connection_mask();
    synchronize_stitch_inputs();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_stitch_inputs");
    run_stitching();
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "after_stitching");
    return finish();
}

void FrameExecution::apply_pending_boundary_edges()
{
    StitchWorkspace& stitch = pipeline_.workspace_.stitch;
    if (stitch.pending_boundary_edges.empty()) {
        return;
    }
    const Stopwatch stopwatch;
    const std::vector<int> changes = stitch.host_boundary.toggle_boundary_edges(
        stitch.pending_boundary_edges
    );
    stitch.pending_boundary_vertex_changes.insert(
        stitch.pending_boundary_vertex_changes.end(), changes.begin(), changes.end()
    );
    stitch.pending_boundary_edges.clear();
    if (stage_timing_ != nullptr) {
        stage_timing_->add_stage(
            "stitch.update_host_boundary_pending",
            stopwatch.elapsed_milliseconds()
        );
    }
}

void FrameExecution::acquire_schedule()
{
    pipeline_.timeline_.log(input_.frame_index, "frame_begin");
    plan_ = pipeline_.scheduler_.begin_frame(input_.frame_index, pipeline_.counts_);
    pipeline_.timeline_.log(
        input_.frame_index,
        "frame_state_ready",
        std::string("prefetched_mesh=") + (plan_.uses_prefetched_mesh ? "1" : "0")
            + " pending_proxy=" + (plan_.uses_pending_proxy ? "1" : "0")
    );
}

void FrameExecution::prepare_scan_prefix()
{
    use_prefetched_scan_ = pipeline_.scheduler_.uses_prefetched_pixel_scan(plan_, input_.frame_index);
    frame_workspace_ = &pipeline_.scheduler_.workspace_for(plan_, use_prefetched_scan_);
    frame_timeline_ = frame_timeline(
        pipeline_.timeline_, pipeline_.options_.overlap_timeline, input_.frame_index
    );
    // With accumulated fill support fixed on, the sequential path builds the
    // scan and its support in one pass. Only pre-scan overlap uses a separate
    // pre-blocker scan prepared by the scheduler.
}

void FrameExecution::prepare_accumulated_depth()
{
    if (plan_.uses_prefetched_mesh) {
        return;
    }
    pipeline_.timeline_.log_cuda(input_.frame_index, "accum_depth_start");
    prepare_cuda_accumulated_depth();
    pipeline_.timeline_.log_cuda(input_.frame_index, "accum_depth_done");
}

void FrameExecution::prepare_cuda_accumulated_depth()
{
    cuda_detail::prepare_accumulated_depth_cuda(
        pipeline_.config_,
        cuda_detail::CudaExecutionContext{pipeline_.workspace_.cuda, stage_timing_},
        cuda_detail::AccumulatedDepthOptions{
            plan_.depth_counts,
            input_.pose,
            &pipeline_.workspace_.accumulated_frame_chunks,
        }
    );
}

void FrameExecution::expose_accumulated_depth()
{
    if (plan_.uses_pending_proxy) {
        pipeline_.timeline_.log_cuda(input_.frame_index, "proxy_paint_start");
    }
    pipeline_.scheduler_.paint_pending_proxy(plan_, input_.frame_index, input_.pose, stage_timing_);
    if (plan_.uses_pending_proxy) {
        pipeline_.timeline_.log_cuda(input_.frame_index, "proxy_paint_done");
    }
    if (plan_.uses_prefetched_mesh) {
        // The proxy worker built this frame with the depth image in the main
        // CUDA workspace. Re-expose that same image so current stitching can
        // build its RI connection mask instead of silently skipping the seam.
        depth_view_ = cuda_detail::AccumulatedDepthView{
            thrust::raw_pointer_cast(pipeline_.workspace_.cuda.accumulated_depth.data()), false
        };
        accumulated_depth_ = &depth_view_;
        return;
    }
    if (!pipeline_.scheduler_.has_accumulated_depth(plan_)) {
        return;
    }
    expose_cuda_accumulated_depth();
    accumulated_depth_ = &depth_view_;
    fill_support_ = &depth_view_;

    const auto& debug = pipeline_.options_.debug.accumulated_depth;
    if (!debug || (debug->max_frames > 0 && pipeline_.dumped_depth_frames_ >= debug->max_frames)) {
        return;
    }
    cuda_detail::timed_cuda_stage_void(stage_timing_, "debug.dump_accumulated_depth", [&] {
        dump_accumulated_depth(
            pipeline_.workspace_.cuda,
            AccumulatedDepthDumpRequest{
                depth_view_,
                static_cast<int>(pipeline_.config_.ri_height),
                static_cast<int>(pipeline_.config_.ri_width),
                input_.frame_index,
                *debug,
            }
        );
    });
    ++pipeline_.dumped_depth_frames_;
}

void FrameExecution::expose_cuda_accumulated_depth()
{
    depth_view_ = cuda_detail::AccumulatedDepthView{
        thrust::raw_pointer_cast(pipeline_.workspace_.cuda.accumulated_depth.data()), false
    };
}

void FrameExecution::build_vertices()
{
    pipeline_.scheduler_.wait_for_pre_scan(use_prefetched_scan_);
    if (plan_.uses_prefetched_mesh) {
        // The worker captured the old-vertex base before the current frame's
        // late vertex appends; rebase its private mesh before consumption.
        pipeline_.scheduler_.rebase_prefetched_mesh(
            pipeline_.counts_.vertex_count,
            stage_timing_
        );
        frame_ = pipeline_.workspace_.prefetch.frame_mesh;
    } else {
        build_filled_pixel_vertices();
    }
    frame_.macro_quads = &frame_workspace_->macro_quads;
    pipeline_.timeline_.log(
        input_.frame_index, "frame_vertices_ready", "new_vertices=" + std::to_string(frame_.new_vertex_count)
    );
    pipeline_.scheduler_.consume_mesh_inputs(plan_, use_prefetched_scan_);
}

void FrameExecution::build_filled_pixel_vertices()
{
    if (use_prefetched_scan_) {
        const cuda_detail::CudaPixelScanPrepass prepass = pipeline_.workspace_.prefetch.pixel_scan;
        cuda_detail::apply_accumulated_fill_support_cuda(
            pipeline_.config_,
            cuda_detail::CudaExecutionContext{*frame_workspace_, stage_timing_, 0, frame_timeline_},
            prepass,
            cuda_detail::ScanBuildOptions{
                pipeline_.counts_.vertex_count,
                nullptr,
                fill_support_,
                true,
                cuda_detail::ProjectedVertexSupportView{
                    thrust::raw_pointer_cast(pipeline_.workspace_.cuda.projected_rows.data()),
                    thrust::raw_pointer_cast(pipeline_.workspace_.cuda.projected_cols.data()),
                    thrust::raw_pointer_cast(pipeline_.workspace_.cuda.projected_ranges.data()),
                    pipeline_.counts_.vertex_count,
                },
            }
        );
        frame_ = cuda_detail::finish_pixel_vertices_from_pre_scan_cuda(
            prepass,
            pipeline_.config_,
            cuda_detail::CudaExecutionContext{*frame_workspace_, stage_timing_, 0, frame_timeline_},
            cuda_detail::ScanBuildOptions{
                pipeline_.counts_.vertex_count, accumulated_depth_
            }
        );
        return;
    }
    frame_ = cuda_detail::scan_and_build_pixel_vertices_cuda(
        input_.points,
        pipeline_.config_,
        cuda_detail::CudaExecutionContext{*frame_workspace_, stage_timing_, 0, frame_timeline_},
        cuda_detail::ScanBuildOptions{
            pipeline_.counts_.vertex_count,
            accumulated_depth_,
            fill_support_,
        }
    );
}

void FrameExecution::build_stitch_connection_mask()
{
    connection_mask_ = nullptr;
    if (accumulated_depth_ == nullptr || frame_workspace_ == nullptr) {
        return;
    }
    const unsigned char* new_valid = thrust::raw_pointer_cast(frame_workspace_->valid.data());
    const unsigned char* overlap_allowed = thrust::raw_pointer_cast(
        frame_workspace_->new_vertex_allowed.data()
    );
    const unsigned char* final_allowed = thrust::raw_pointer_cast(
        frame_workspace_->final_new_allowed.data()
    );
    const unsigned char* thin_removed = thrust::raw_pointer_cast(
        frame_workspace_->ri_thin_removed.data()
    );
    cuda_detail::build_stitch_connection_mask(
        cuda_detail::StitchConnectionMaskRequest{
            cuda_detail::CudaExecutionContext{*frame_workspace_, stage_timing_},
            *accumulated_depth_,
            new_valid,
            nullptr,
            overlap_allowed,
            final_allowed,
            thin_removed,
            static_cast<int>(pipeline_.config_.ri_height),
            static_cast<int>(pipeline_.config_.ri_width),
            pipeline_.config_.stitch_connection_frontier_expand_radius,
        }
    );
}

void FrameExecution::materialize_stitch_connection_mask()
{
    if (frame_workspace_ == nullptr || accumulated_depth_ == nullptr) {
        return;
    }
    StitchConnectionMask& mask = frame_workspace_->host_stitch_connection_mask;
    cuda_detail::materialize_stitch_connection_mask(
        cuda_detail::StitchConnectionMaskMaterializeRequest{
            cuda_detail::CudaExecutionContext{*frame_workspace_, stage_timing_},
            static_cast<int>(pipeline_.config_.ri_height),
            static_cast<int>(pipeline_.config_.ri_width),
            mask,
        }
    );
    connection_mask_ = &mask;
}

void FrameExecution::dump_overlap_debug()
{
    const auto& debug = pipeline_.options_.debug.overlap;
    const bool limit_reached = debug && debug->max_frames > 0
        && pipeline_.dumped_overlap_frames_ >= debug->max_frames;
    if (!debug || accumulated_depth_ == nullptr || limit_reached || frame_workspace_ == nullptr) {
        return;
    }
    const unsigned char* new_valid = thrust::raw_pointer_cast(frame_workspace_->valid.data());
    const unsigned char* overlap_allowed = thrust::raw_pointer_cast(
        frame_workspace_->new_vertex_allowed.data()
    );
    const unsigned char* final_allowed = thrust::raw_pointer_cast(
        frame_workspace_->final_new_allowed.data()
    );
    const unsigned char* thin_removed = thrust::raw_pointer_cast(
        frame_workspace_->ri_thin_removed.data()
    );
    cuda_detail::timed_cuda_stage_void(stage_timing_, "debug.dump_overlap", [&] {
        cuda_pipeline::dump_overlap_debug(
            OverlapDebugRequest{
                *frame_workspace_,
                *accumulated_depth_,
                new_valid,
            nullptr,
                overlap_allowed,
                final_allowed,
                thin_removed,
                static_cast<int>(pipeline_.config_.ri_height),
                static_cast<int>(pipeline_.config_.ri_width),
                input_.frame_index,
                pipeline_.config_.overlap_neighbor_euclidean_tol,
                *debug,
            }
        );
    });
    ++pipeline_.dumped_overlap_frames_;
}

void FrameExecution::dump_ri_thin_filter()
{
    const auto& debug = pipeline_.options_.debug.ri_thin_filter;
    const bool limit_reached = debug && debug->max_frames > 0
        && pipeline_.dumped_ri_thin_filter_frames_ >= debug->max_frames;
    if (!debug || frame_workspace_ == nullptr || limit_reached) {
        return;
    }
    cuda_detail::timed_cuda_stage_void(stage_timing_, "debug.dump_ri_thin_filter", [&] {
        cuda_pipeline::dump_ri_thin_filter(
            *frame_workspace_,
            static_cast<int>(pipeline_.config_.ri_height),
            static_cast<int>(pipeline_.config_.ri_width),
            input_.frame_index,
            *debug
        );
    });
    ++pipeline_.dumped_ri_thin_filter_frames_;
}

void FrameExecution::dump_filled_ri()
{
    const auto& debug = pipeline_.options_.debug.filled_ri;
    const bool limit_reached = debug && debug->max_frames > 0
        && pipeline_.dumped_filled_ri_frames_ >= debug->max_frames;
    if (!debug || limit_reached) {
        return;
    }
    cuda_detail::timed_cuda_stage_void(stage_timing_, "debug.dump_filled_ri", [&] {
        cuda_pipeline::dump_filled_ri(
            *frame_workspace_,
            static_cast<int>(pipeline_.config_.ri_height),
            static_cast<int>(pipeline_.config_.ri_width),
            input_.frame_index,
            *debug
        );
    });
    ++pipeline_.dumped_filled_ri_frames_;
}

void FrameExecution::generate_dense_mesh()
{
    device_faces_ = &pipeline_.workspace_.prefetch.cuda.faces;
    if (!plan_.uses_prefetched_mesh) {
        pipeline_.timeline_.log_cuda(input_.frame_index, "dense_mesh_start");
        device_faces_ = &cuda_detail::coarsened_dense_mesh_cuda(
            frame_,
            pipeline_.config_,
            cuda_detail::CudaExecutionContext{*frame_workspace_, stage_timing_},
            cuda_detail::DenseMeshOptions{
                accumulated_depth_,
            }
        );
        pipeline_.timeline_.log_cuda(
            input_.frame_index, "dense_mesh_done", 0, "faces=" + std::to_string(device_faces_->size())
        );
    }
    pipeline_.timeline_.log(
        input_.frame_index, "dense_mesh_ready", "faces=" + std::to_string(device_faces_->size())
    );
}

void FrameExecution::append_dense_mesh()
{
    cuda_detail::CudaWorkspace& frame_workspace = frame_workspace_ != nullptr
        ? *frame_workspace_ : pipeline_.workspace_.cuda;
    cuda_detail::compact_current_frame_vertices_cuda(
        frame_,
        *device_faces_,
        cuda_detail::CudaExecutionContext{frame_workspace, stage_timing_}
    );
    new_face_count_ = device_faces_->size();
    old_vertex_count_ = pipeline_.counts_.vertex_count;
    old_face_count_ = pipeline_.counts_.face_count;
    pipeline_.timeline_.log_cuda(input_.frame_index, "append_mesh_start");
    cuda_detail::append_current_frame_to_accumulation_cuda(
        pipeline_.counts_,
        cuda_detail::CudaExecutionContext{pipeline_.workspace_.cuda, stage_timing_},
        cuda_detail::AccumulationAppendInput{frame_, *device_faces_, input_.pose}
    );
    launch_stitch_normal_accumulation();
    pipeline_.timeline_.log_cuda(input_.frame_index, "append_mesh_done");
    pipeline_.timeline_.log(input_.frame_index, "new_mesh_appended");
}

void FrameExecution::launch_stitch_normal_accumulation()
{
    launch_device_face_normal_sums(
        DeviceNormalAccumulationRequest{
            pipeline_.workspace_.cuda,
            *device_faces_,
            old_vertex_count_,
            frame_.new_vertex_count,
            pipeline_.workspace_.stitch.host_vertex_normal_sums_global,
            pipeline_.workspace_.stitch.host_current_vertex_normal_sums,
            stage_timing_,
            0,
        }
    );
}

void FrameExecution::extract_new_boundary()
{
    std::vector<std::array<int, 2>>& edges = pipeline_.workspace_.stitch.new_boundary_edges;
    edges.clear();
    if (new_face_count_ == 0) {
        return;
    }
    pipeline_.timeline_.log_cuda(input_.frame_index, "stitch.extract_new_boundary_start");
    cuda_detail::timed_cuda_stage_void(stage_timing_, "stitch.extract_new_boundary", [&] {
        batch_boundary_edges_sorted_cuda(
            pipeline_.workspace_.cuda,
            old_face_count_,
            static_cast<int>(new_face_count_),
            edges
        );
    });
    const std::string edge_count = "edges=" + std::to_string(edges.size());
    pipeline_.timeline_.log_cuda(
        input_.frame_index, "stitch.extract_new_boundary_done", 0, edge_count
    );
    pipeline_.timeline_.log(input_.frame_index, "stitch.new_boundary_ready", edge_count);
}

void FrameExecution::synchronize_stitch_inputs()
{
    pipeline_.timeline_.log(
        input_.frame_index,
        "sync_host_vertices_start",
        "new_vertices=" + std::to_string(frame_.new_vertex_count)
            + " new_faces=" + std::to_string(new_face_count_)
    );
    const Stopwatch handoff_stopwatch;
    sync_new_host_vertices(
        pipeline_.workspace_.cuda,
        pipeline_.workspace_.stitch.host_vertices_global,
        old_vertex_count_,
        frame_.new_vertex_count,
        stage_timing_
    );
    materialize_device_face_normal_sums(
        DeviceNormalAccumulationRequest{
            pipeline_.workspace_.cuda,
            *device_faces_,
            old_vertex_count_,
            frame_.new_vertex_count,
            pipeline_.workspace_.stitch.host_vertex_normal_sums_global,
            pipeline_.workspace_.stitch.host_current_vertex_normal_sums,
            stage_timing_,
            0,
        }
    );
    pipeline_.timeline_.log(input_.frame_index, "sync_host_vertices_done");
    pipeline_.timeline_.log(input_.frame_index, "new_vertices_synced_host");
    register_accumulated_frame_chunk();
    if (stage_timing_ != nullptr) {
        stage_timing_->add_stage(
            "stitch.synchronize_stitch_inputs.total",
            handoff_stopwatch.elapsed_milliseconds()
        );
    }
}

void FrameExecution::register_accumulated_frame_chunk()
{
    if (new_face_count_ == 0) {
        return;
    }
    const Stopwatch stopwatch;
    auto& chunks = pipeline_.workspace_.accumulated_frame_chunks;
    accumulated_frame_chunk_index_ = chunks.size();
    has_accumulated_frame_chunk_ = true;
    chunks.push_back(make_accumulated_frame_chunk(
        pipeline_.workspace_.stitch.host_vertices_global,
        old_vertex_count_,
        frame_.new_vertex_count,
        old_face_count_,
        static_cast<int>(new_face_count_)
    ));
    if (stage_timing_ != nullptr) {
        stage_timing_->add_stage(
            "stitch.synchronize_stitch_inputs.register_frame_chunk",
            stopwatch.elapsed_milliseconds()
        );
    }
}

void FrameExecution::run_stitching()
{
    std::optional<Mat4f> next_pose;
    if (frame_position_ + 1 < pipeline_.frames_.size()) {
        next_pose = pipeline_.frames_[frame_position_ + 1].pose;
    }
    pipeline_.timeline_.log(input_.frame_index, "boundary_stitching_start");
    stitch_result_ = pipeline_.stitcher_.process(
        StitchFrameInput{
            frame_position_,
            input_.frame_index,
            old_vertex_count_,
            old_face_count_,
            static_cast<int>(new_face_count_),
            input_.pose,
            std::move(next_pose),
            connection_mask_,
            frame_workspace_ != nullptr
                ? thrust::raw_pointer_cast(frame_workspace_->stitch_connection_mask.data()) : nullptr,
            frame_workspace_ != nullptr
                ? thrust::raw_pointer_cast(frame_workspace_->filled_ranges.data()) : nullptr,
        },
        pipeline_.counts_,
        stage_timing_
    );
    finalize_accumulated_frame_chunk();
    pipeline_.scheduler_.publish_prefetched_frame_append(
        PrefetchedFrameAppend{
            pipeline_.counts_,
            old_vertex_count_ + frame_.new_vertex_count,
            pipeline_.counts_.vertex_count - old_vertex_count_ - frame_.new_vertex_count,
            old_face_count_ + static_cast<int>(new_face_count_),
            pipeline_.counts_.face_count - old_face_count_ - static_cast<int>(new_face_count_),
        }
    );
    pipeline_.timeline_.log(
        input_.frame_index,
        "boundary_stitching_done",
        "stitch_faces=" + std::to_string(stitch_result_.stitch_faces)
    );
    pipeline_.scheduler_.publish_proxy(stitch_result_, frame_position_, plan_.uses_pending_proxy);
}

void FrameExecution::finalize_accumulated_frame_chunk()
{
    if (!has_accumulated_frame_chunk_) {
        return;
    }
    auto& chunk = pipeline_.workspace_.accumulated_frame_chunks[accumulated_frame_chunk_index_];
    chunk.vertex_count = std::max(0, pipeline_.counts_.vertex_count - chunk.first_vertex);
    chunk.face_count = std::max(0, pipeline_.counts_.face_count - chunk.first_face);
}

FrameProcessingStats FrameExecution::finish()
{
    pipeline_.timeline_.log(input_.frame_index, "frame_main_stream_sync_start");
    const Stopwatch finish_sync_stopwatch;
    // The prefetch worker owns a separate non-blocking stream.  Its readiness
    // is consumed through scheduler events, so a frame-local completion must
    // not wait for the next frame's whole GPU prefix here.
    CUDA_CHECK(cudaStreamSynchronize(0));
    const double finish_sync_milliseconds = finish_sync_stopwatch.elapsed_milliseconds();
    if (stage_timing_ != nullptr) {
        stage_timing_->add_stage(
            "pipeline.frame_finish_main_stream_sync",
            finish_sync_milliseconds
        );
    }
    pipeline_.timeline_.log(
        input_.frame_index,
        "frame_main_stream_sync_done",
        "ms=" + std::to_string(finish_sync_milliseconds)
    );
    record_memory_snapshot(pipeline_.workspace_, input_.frame_index, "frame_end");
    timing_.total_milliseconds = total_stopwatch_.elapsed_milliseconds();
    pipeline_.timeline_.log(
        input_.frame_index, "frame_end", "ms=" + std::to_string(timing_.total_milliseconds)
    );
    return FrameProcessingStats{
        input_.frame_index,
        input_.points.size(),
        static_cast<std::size_t>(frame_.new_vertex_count),
        new_face_count_,
        stitch_result_.stitch_faces,
        stitch_result_.hole_faces,
        static_cast<std::size_t>(pipeline_.counts_.vertex_count),
        static_cast<std::size_t>(pipeline_.counts_.face_count),
        std::move(timing_),
        stitch_result_.proxy,
    };
}

}  // namespace ri_meshing::cuda_pipeline
