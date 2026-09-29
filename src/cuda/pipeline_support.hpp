#pragma once

#include "cuda_utils.hpp"
#include "ri_meshing/cuda_meshing.hpp"
#include "ri_meshing/timing.hpp"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

namespace ri_meshing::cuda_pipeline {

inline PipelineTiming* detailed_timing_or_null(TimingMode timing_mode, PipelineTiming& timing)
{
#if defined(RI_MESHING_ENABLE_DETAILED_TIMING)
    if (timing_mode == TimingMode::summary) {
        return nullptr;
    }
    timing.synchronize_cuda = timing_mode == TimingMode::detailed;
    return &timing;
#else
    (void)timing_mode;
    (void)timing;
    return nullptr;
#endif
}

class OverlapTimeline {
public:
    OverlapTimeline(bool enabled, bool sync_cuda)
        : enabled_(enabled)
        , sync_cuda_(sync_cuda)
    {
    }

    void log(int frame_index, std::string_view event, std::string_view detail = {})
    {
        if (!enabled_) {
            return;
        }
        const std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << "overlap.timeline"
                  << " t_ms=" << std::fixed << std::setprecision(3) << stopwatch_.elapsed_milliseconds()
                  << " frame=" << frame_index
                  << " event=" << event;
        if (!detail.empty()) {
            std::cerr << ' ' << detail;
        }
        std::cerr << '\n';
    }

    void log_cuda(int frame_index, std::string_view event, cudaStream_t stream = 0, std::string_view detail = {})
    {
        if (!enabled_) {
            return;
        }
        if (sync_cuda_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        log(frame_index, event, detail);
    }

private:
    bool enabled_{};
    bool sync_cuda_{};
    Stopwatch stopwatch_;
    std::mutex mutex_;
};

}  // namespace ri_meshing::cuda_pipeline
