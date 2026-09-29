#pragma once

#include "cuda_workflow.hpp"

#include <cstdint>
#include <vector>

namespace ri_meshing::cuda_detail {

struct StitchConnectionMaskRequest {
    CudaExecutionContext execution;
    AccumulatedDepthView accumulated_depth;
    const unsigned char* new_valid{};
    const int* effective_new_vim{};
    const unsigned char* overlap_allowed{};
    const unsigned char* final_allowed{};
    const unsigned char* thin_removed{};
    int height{};
    int width{};
    int frontier_expand_radius{};
};

void build_stitch_connection_mask(StitchConnectionMaskRequest request);

struct StitchConnectionMaskMaterializeRequest {
    CudaExecutionContext execution;
    int height{};
    int width{};
    StitchConnectionMask& host_output;
};

void materialize_stitch_connection_mask(
    StitchConnectionMaskMaterializeRequest request
);

}  // namespace ri_meshing::cuda_detail
