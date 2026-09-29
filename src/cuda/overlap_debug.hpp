#pragma once

#include "pipeline_support.hpp"
#include "cuda_workflow.hpp"

namespace ri_meshing::cuda_pipeline {

struct OverlapDebugRequest {
    cuda_detail::CudaWorkspace& workspace;
    cuda_detail::AccumulatedDepthView depth;
    const unsigned char* new_valid;
    const int* effective_new_vim;
    const unsigned char* overlap_allowed;
    const unsigned char* final_allowed;
    const unsigned char* thin_removed;
    int height;
    int width;
    int frame_index;
    MeshScalar neighbor_tolerance;
    const OverlapDebugDump& debug_dump;
};

void dump_overlap_debug(OverlapDebugRequest request);

}  // namespace ri_meshing::cuda_pipeline
