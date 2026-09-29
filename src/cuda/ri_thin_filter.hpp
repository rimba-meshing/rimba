#pragma once

#include "cuda_workflow.hpp"

namespace ri_meshing::cuda_detail {

struct RiThinFilterRequest {
    CudaExecutionContext execution;
    const unsigned char* input{};
    int height{};
    int width{};
};

void apply_ri_thin_filter(RiThinFilterRequest request);

}  // namespace ri_meshing::cuda_detail
