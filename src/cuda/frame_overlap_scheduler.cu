#include "frame_overlap_scheduler.hpp"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"
#include "pipeline_support.hpp"
#include "stitch_gpu_helpers.hpp"

#include <thrust/device_ptr.h>

#include <algorithm>
#include <future>
#include <memory>
#include <utility>

namespace ri_meshing::cuda_pipeline {
namespace {

cuda_detail::StageTimelineCallback make_timeline_callback(
    OverlapTimeline* timeline,
    int frame_index,
    cudaStream_t stream
)
{
    if (timeline == nullptr) {
        return {};
    }
    return [timeline, frame_index, stream](const char* event) {
        timeline->log_cuda(frame_index, event, stream);
    };
}

void log_event(OverlapTimeline* timeline, int frame_index, const char* event)
{
    if (timeline != nullptr) {
        timeline->log(frame_index, event);
    }
}

template <typename T>
void snapshot_device_values(
    thrust::device_vector<T>& destination,
    const thrust::device_vector<T>& source,
    std::size_t count,
    cudaStream_t stream
)
{
    cuda_detail::resize_with_headroom(destination, count);
    cuda_detail::copy_device_to_device_async(
        thrust::raw_pointer_cast(destination.data()),
        thrust::raw_pointer_cast(source.data()),
        count,
        stream
    );
}

void snapshot_proxy_inputs(
    cuda_detail::CudaWorkspace& source,
    cuda_detail::CudaWorkspace& destination,
    int vertex_count,
    cudaStream_t stream
)
{
    snapshot_device_values(
        destination.accumulated_depth,
        source.accumulated_depth,
        source.accumulated_depth.size(),
        stream
    );
    const std::size_t projection_count = static_cast<std::size_t>(
        std::max(0, vertex_count)
    );
    snapshot_device_values(
        destination.projected_rows,
        source.projected_rows,
        std::min(projection_count, source.projected_rows.size()),
        stream
    );
    snapshot_device_values(
        destination.projected_cols,
        source.projected_cols,
        std::min(projection_count, source.projected_cols.size()),
        stream
    );
    snapshot_device_values(
        destination.projected_ranges,
        source.projected_ranges,
        std::min(projection_count, source.projected_ranges.size()),
        stream
    );
}

}  // namespace

FrameOverlapScheduler::FrameOverlapScheduler(
    OverlapSchedulerResources resources,
    OverlapSchedulerSettings settings
)
    : workspace_(resources.workspace)
    , frames_(resources.frames)
    , config_(resources.config)
    , overlap_pre_scan_(settings.overlap_pre_scan)
    , stitch_proxy_blocker_(settings.stitch_proxy_blocker)
    , stitch_proxy_kernel_timing_(settings.stitch_proxy_kernel_timing)
    , timeline_(resources.timeline)
{
}

FrameExecutionPlan FrameOverlapScheduler::begin_frame(
    int frame_index,
    cuda_detail::CudaAccumulationCounts counts
)
{
    FrameExecutionPlan plan;
    plan.uses_prefetched_mesh = overlap_pre_scan_ && stitch_proxy_blocker_
        && workspace_.overlap.has_prefetched_frame_mesh && workspace_.overlap.prefetched_frame_mesh_index == frame_index;
    if (plan.uses_prefetched_mesh) {
        log_event(timeline_, frame_index, "prefetch.consume_wait_begin");
        wait_prefetched_frame_mesh(workspace_);
        log_event(timeline_, frame_index, "prefetch.consume_wait_done");
    }
    plan.uses_pending_proxy = stitch_proxy_blocker_ && !plan.uses_prefetched_mesh
        && workspace_.overlap.has_pending_stitch_proxy
        && workspace_.overlap.pending_stitch_proxy_frame_index == frame_index;
    plan.depth_counts = counts;
    if (plan.uses_pending_proxy) {
        plan.depth_counts.face_count = std::min(counts.face_count, workspace_.overlap.pending_stitch_proxy_base_face_count);
    }
    return plan;
}

void FrameOverlapScheduler::rebase_prefetched_mesh(
    int new_old_vertex_count,
    PipelineTiming* timing
)
{
    if (!workspace_.overlap.has_prefetched_frame_mesh) {
        return;
    }
    cuda_detail::rebase_prefetched_frame_cuda(
        workspace_.prefetch.frame_mesh,
        workspace_.prefetch.cuda.faces,
        new_old_vertex_count,
        cuda_detail::CudaExecutionContext{workspace_.prefetch.cuda, timing}
    );
}

void FrameOverlapScheduler::paint_pending_proxy(
    const FrameExecutionPlan& plan,
    int frame_index,
    const Mat4f& pose,
    PipelineTiming* timing
)
{
    if (plan.uses_pending_proxy) {
        paint_stitch_proxy_edges_into_accumulated_cuda(
            workspace_.cuda,
            workspace_.overlap.pending_stitch_proxy_edges,
            pose,
            ProxyPaintContext{config_, timing, 0, stitch_proxy_kernel_timing_, frame_index}
        );
    }
}

bool FrameOverlapScheduler::has_accumulated_depth(const FrameExecutionPlan& plan) const
{
    if (plan.uses_prefetched_mesh) {
        return false;
    }
    const bool needed_for_overlap = plan.depth_counts.face_count > 0 || plan.uses_pending_proxy;
    const bool needed_for_fill_support = plan.depth_counts.vertex_count > 0;
    return needed_for_overlap || needed_for_fill_support;
}

bool FrameOverlapScheduler::uses_prefetched_pixel_scan(const FrameExecutionPlan& plan, int frame_index) const
{
    return !plan.uses_prefetched_mesh && overlap_pre_scan_
        && workspace_.overlap.has_prefetched_pixel_scan && workspace_.overlap.prefetched_frame_index == frame_index;
}

void FrameOverlapScheduler::wait_for_pre_scan(bool uses_prefetched_scan)
{
    if (uses_prefetched_scan) {
        wait_prefetched_pixel_scan(workspace_);
    }
}

cuda_detail::CudaWorkspace& FrameOverlapScheduler::workspace_for(
    const FrameExecutionPlan& plan,
    bool uses_prefetched_scan
)
{
    return (plan.uses_prefetched_mesh || uses_prefetched_scan) ? workspace_.prefetch.cuda : workspace_.cuda;
}

void FrameOverlapScheduler::consume_mesh_inputs(const FrameExecutionPlan& plan, bool uses_prefetched_scan)
{
    if (plan.uses_prefetched_mesh) {
        workspace_.overlap.has_prefetched_frame_mesh = false;
        workspace_.overlap.has_prefetched_pixel_scan = false;
        workspace_.overlap.pending_stitch_proxy_edges = {};
        workspace_.overlap.has_pending_stitch_proxy = false;
    } else if (uses_prefetched_scan) {
        workspace_.overlap.has_prefetched_pixel_scan = false;
    }
}

void FrameOverlapScheduler::start_pre_scan(std::size_t frame_position)
{
    if (!overlap_pre_scan_ || frame_position + 1 >= frames_.size() || workspace_.overlap.has_prefetched_pixel_scan) {
        return;
    }
    wait_prefetched_pixel_scan(workspace_);
    const CudaFrameInput& next_frame = frames_[frame_position + 1];
    workspace_.overlap.prefetched_frame_index = next_frame.frame_index;
    workspace_.overlap.has_prefetched_pixel_scan = true;
    workspace_.overlap.prefetched_pixel_scan_task = std::async(std::launch::async, [this, points = &next_frame.points, frame_index = next_frame.frame_index] {
        cudaStream_t stream{};
        try {
            // cudaStreamCreateWithFlags initializes the CUDA runtime for this
            // worker.  cudaFree(nullptr) used to be called here as an
            // implicit initialization fence on every prefetched frame; it
            // adds a device-wide synchronization without providing an
            // ordering guarantee that the explicit task/event waits do not
            // already provide.
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            workspace_.prefetch.pixel_scan = cuda_detail::scan_pixel_vertices_pre_blocker_cuda(
                *points,
                config_,
                cuda_detail::CudaExecutionContext{
                    workspace_.prefetch.cuda,
                    nullptr,
                    stream,
                    make_timeline_callback(timeline_, frame_index, stream),
                },
                cuda_detail::ScanBuildOptions{
                    0,
                    nullptr,
                    nullptr,
                    true,
                }
            );
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamDestroy(stream));
        } catch (...) {
            if (stream != nullptr) {
                static_cast<void>(cudaStreamDestroy(stream));
            }
            throw;
        }
    });
}

bool FrameOverlapScheduler::supports_proxy_prefetch() const noexcept
{
    return overlap_pre_scan_ && stitch_proxy_blocker_;
}

bool FrameOverlapScheduler::start_proxy_prefetch(ProxyPrefetchRequest request)
{
    const bool has_proxy_source = !request.proxy_edges.empty();
    const bool cannot_start = !supports_proxy_prefetch()
        || request.frame_position + 1 >= frames_.size()
        || !workspace_.overlap.has_prefetched_pixel_scan || workspace_.overlap.has_prefetched_frame_mesh
        || !has_proxy_source;
    if (cannot_start) {
        return false;
    }
    wait_prefetched_pixel_scan(workspace_);
    wait_prefetched_frame_mesh(workspace_);
    const CudaFrameInput& source_frame = frames_[request.frame_position];
    const CudaFrameInput& next_frame = frames_[request.frame_position + 1];
    const cuda_detail::CudaAccumulationCounts depth_counts{
        request.counts.vertex_count,
        std::min(request.counts.face_count, request.proxy_base_face_count)
    };
    workspace_.overlap.prefetched_frame_mesh_index = next_frame.frame_index;
    workspace_.overlap.has_prefetched_frame_mesh = true;
    auto prefix_recorded = std::make_shared<std::promise<void>>();
    workspace_.overlap.prefetched_prefix_read_recorded = prefix_recorded->get_future().share();
    CUDA_CHECK(cudaEventCreateWithFlags(&workspace_.overlap.prefetched_prefix_read_done_event, cudaEventDisableTiming));
    workspace_.overlap.has_prefetched_prefix_read_done_event = true;
    CUDA_CHECK(cudaEventCreateWithFlags(&workspace_.overlap.prefetched_frame_mesh_ready_event, cudaEventDisableTiming));
    workspace_.overlap.has_prefetched_frame_mesh_ready_event = true;
    auto frame_append = std::make_shared<PrefetchedFrameAppend>();
    auto frame_append_promise = std::make_shared<std::promise<void>>();
    workspace_.overlap.prefetched_frame_append = frame_append;
    workspace_.overlap.prefetched_frame_append_promise = frame_append_promise;
    ProxyPrefetchJob job{
        depth_counts,
        workspace_.accumulated_frame_chunks,
        next_frame.pose,
        request.proxy_edges,
        source_frame.pose,
        source_frame.frame_index,
        next_frame.frame_index,
        stitch_proxy_kernel_timing_,
        workspace_.overlap.prefetched_prefix_read_done_event,
        workspace_.overlap.prefetched_frame_mesh_ready_event,
        std::move(prefix_recorded),
        std::move(frame_append),
        std::move(frame_append_promise),
        workspace_.overlap.prefetched_frame_append_promise->get_future().share(),
    };
    workspace_.overlap.prefetched_frame_mesh_task = std::async(
        std::launch::async,
        [this, job = std::move(job)]() mutable { run_proxy_prefetch(std::move(job)); }
    );
    return true;
}

// This worker stays linear because it owns the ordering between the shared
// depth snapshot, the private proxy build, and the exact post-dense append.
void FrameOverlapScheduler::run_proxy_prefetch(ProxyPrefetchJob job)
{
    cudaStream_t stream{};
    bool prefix_published = false;
    bool frame_append_consumed = false;
    try {
        log_event(timeline_, job.frame_index, "proxy_worker_start");
        // Stream creation performs the required runtime initialization.  Do
        // not use cudaFree(nullptr) as a per-frame fence: this worker is
        // ordered through the pre-scan future and CUDA events below.
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (timeline_ != nullptr) timeline_->log_cuda(job.frame_index, "proxy_depth_start", stream);
        cuda_detail::prepare_accumulated_depth_cuda(
            config_,
            cuda_detail::CudaExecutionContext{workspace_.cuda, nullptr, stream},
            cuda_detail::AccumulatedDepthOptions{
                job.depth_counts,
                job.pose,
                &job.frame_chunks,
            }
        );
        if (timeline_ != nullptr) {
            timeline_->log_cuda(job.frame_index, "proxy_depth_done", stream);
            timeline_->log_cuda(job.frame_index, "proxy_paint_start", stream);
        }
        paint_stitch_proxy_edges_into_accumulated_cuda(
            workspace_.cuda,
            job.proxy_edges,
            job.pose,
            ProxyPaintContext{
                config_, nullptr, stream, job.measure_kernel_timing, job.source_frame_index
            }
        );
        if (timeline_ != nullptr) timeline_->log_cuda(job.frame_index, "proxy_paint_done", stream);
        snapshot_proxy_inputs(
            workspace_.cuda,
            workspace_.prefetch.cuda,
            job.depth_counts.vertex_count,
            stream
        );
        if (timeline_ != nullptr) {
            timeline_->log_cuda(job.frame_index, "proxy_depth_snapshot_done", stream);
        }
        CUDA_CHECK(cudaEventRecord(job.prefix_event, stream));
        job.prefix_recorded->set_value();
        prefix_published = true;
        log_event(timeline_, job.frame_index, "proxy_worker_prefix_recorded");
        const cuda_detail::AccumulatedDepthView depth{
            thrust::raw_pointer_cast(workspace_.prefetch.cuda.accumulated_depth.data()), false
        };
        // In proxy mode the blocker is intentionally approximate. Reuse
        // that same approximation as accumulated fill support so the
        // whole vertex build remains inside the asynchronous worker.
        // The normal pre-scan already filled the current scan and tracked
        // anchors; this deferred pass adds proxy-supported ranges before
        // overlap rejection and dense meshing.
        cuda_detail::apply_accumulated_fill_support_cuda(
            config_,
            cuda_detail::CudaExecutionContext{
                workspace_.prefetch.cuda,
                nullptr,
                stream,
                make_timeline_callback(timeline_, job.frame_index, stream),
            },
            workspace_.prefetch.pixel_scan,
            cuda_detail::ScanBuildOptions{
                job.depth_counts.vertex_count,
                nullptr,
                &depth,
                true,
                cuda_detail::ProjectedVertexSupportView{
                    thrust::raw_pointer_cast(workspace_.prefetch.cuda.projected_rows.data()),
                    thrust::raw_pointer_cast(workspace_.prefetch.cuda.projected_cols.data()),
                    thrust::raw_pointer_cast(workspace_.prefetch.cuda.projected_ranges.data()),
                    job.depth_counts.vertex_count,
                },
            }
        );
        workspace_.prefetch.frame_mesh = cuda_detail::finish_pixel_vertices_from_pre_scan_cuda(
            workspace_.prefetch.pixel_scan,
            config_,
            cuda_detail::CudaExecutionContext{
                workspace_.prefetch.cuda,
                nullptr,
                stream,
                make_timeline_callback(timeline_, job.frame_index, stream),
            },
            cuda_detail::ScanBuildOptions{job.depth_counts.vertex_count, &depth}
        );
        workspace_.prefetch.frame_mesh.macro_quads = &workspace_.prefetch.cuda.macro_quads;
        if (timeline_ != nullptr) timeline_->log_cuda(job.frame_index, "dense_mesh_start", stream);
        cuda_detail::coarsened_dense_mesh_cuda(
            workspace_.prefetch.frame_mesh,
            config_,
            cuda_detail::CudaExecutionContext{workspace_.prefetch.cuda, nullptr, stream},
            cuda_detail::DenseMeshOptions{&depth}
        );
        if (timeline_ != nullptr) timeline_->log_cuda(job.frame_index, "dense_mesh_done", stream);
        // The overlap decision above intentionally used the proxy snapshot.
        // Once dense meshing is complete, append the exact faces produced by
        // the source frame to the main depth image for the next consumer.
        job.frame_append_ready.get();
        frame_append_consumed = true;
        if (job.frame_append != nullptr && job.frame_append->face_count > 0) {
            if (timeline_ != nullptr) {
                timeline_->log_cuda(job.frame_index, "proxy_post_faces_start", stream);
            }
            cuda_detail::append_accumulated_depth_cuda(
                config_,
                cuda_detail::CudaExecutionContext{workspace_.cuda, nullptr, stream},
                cuda_detail::AccumulatedDepthAppendOptions{
                    job.frame_append->counts,
                    job.pose,
                    job.frame_append->first_vertex,
                    job.frame_append->vertex_count,
                    job.frame_append->first_face,
                    job.frame_append->face_count,
                }
            );
            if (timeline_ != nullptr) {
                timeline_->log_cuda(job.frame_index, "proxy_post_faces_done", stream);
            }
        }
        CUDA_CHECK(cudaEventRecord(job.ready_event, stream));
        log_event(timeline_, job.frame_index, "proxy_worker_mesh_recorded");
        CUDA_CHECK(cudaStreamDestroy(stream));
    } catch (...) {
        if (stream != nullptr) static_cast<void>(cudaStreamDestroy(stream));
        if (!prefix_published) job.prefix_recorded->set_exception(std::current_exception());
        if (prefix_published && !frame_append_consumed && job.frame_append_promise != nullptr) {
            try {
                job.frame_append_promise->set_exception(std::current_exception());
            } catch (...) {
                // The main thread may have published the append metadata at
                // the same time the worker failed. In that case the future
                // is already resolved and there is nothing left to signal.
            }
        }
        throw;
    }
}

void FrameOverlapScheduler::wait_for_proxy_prefix()
{
    if (!workspace_.overlap.has_prefetched_frame_mesh) {
        return;
    }
    log_event(timeline_, workspace_.overlap.prefetched_frame_mesh_index, "prefix_wait_begin");
    wait_prefetched_prefix_read(workspace_);
    log_event(timeline_, workspace_.overlap.prefetched_frame_mesh_index, "prefix_wait_done");
}

void FrameOverlapScheduler::publish_proxy(
    StitchAppendResult& result,
    std::size_t frame_position,
    bool consumed_pending_proxy
)
{
    if (stitch_proxy_blocker_ && frame_position + 1 < frames_.size()
        && !result.proxy_edges.empty()) {
        workspace_.overlap.pending_stitch_proxy_edges = std::move(result.proxy_edges);
        workspace_.overlap.pending_stitch_proxy_frame_index = frames_[frame_position + 1].frame_index;
        workspace_.overlap.pending_stitch_proxy_base_face_count = result.proxy_base_face_count;
        workspace_.overlap.has_pending_stitch_proxy = true;
    } else if (consumed_pending_proxy) {
        workspace_.overlap.pending_stitch_proxy_edges = {};
        workspace_.overlap.has_pending_stitch_proxy = false;
    }
}

void FrameOverlapScheduler::publish_prefetched_frame_append(PrefetchedFrameAppend append)
{
    if (workspace_.overlap.prefetched_frame_append == nullptr
        || workspace_.overlap.prefetched_frame_append_promise == nullptr) {
        return;
    }
    *workspace_.overlap.prefetched_frame_append = append;
    workspace_.overlap.prefetched_frame_append_promise->set_value();
    workspace_.overlap.prefetched_frame_append.reset();
    workspace_.overlap.prefetched_frame_append_promise.reset();
}

}  // namespace ri_meshing::cuda_pipeline
