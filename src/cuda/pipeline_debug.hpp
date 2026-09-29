#pragma once

#include "cuda_workflow.hpp"

namespace ri_meshing::cuda_pipeline {

struct AccumulatedDepthDumpRequest {
    cuda_detail::AccumulatedDepthView depth;
    int height{};
    int width{};
    int frame_index{};
    const AccumDepthDebugDump& debug_dump;
};

void dump_accumulated_depth(
    cuda_detail::CudaWorkspace& workspace,
    const AccumulatedDepthDumpRequest& request
);
void dump_filled_ri(
    cuda_detail::CudaWorkspace& workspace,
    int height,
    int width,
    int frame_index,
    const FilledRiDebugDump& debug_dump
);
void dump_ri_thin_filter(
    cuda_detail::CudaWorkspace& workspace,
    int height,
    int width,
    int frame_index,
    const RiThinFilterDebugDump& debug_dump
);

}  // namespace ri_meshing::cuda_pipeline
