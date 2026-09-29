#pragma once

#include "cuda_workflow.hpp"

namespace ri_meshing::cuda_detail {

struct DenseCandidateRequest {
    const CudaRiFrame& frame;
    const SparseMeshingConfig& config;
    CudaExecutionContext execution;
    DenseMeshOptions options;
};

struct DenseIsolationRequest {
    const CudaRiFrame& frame;
    const SparseMeshingConfig& config;
    CudaExecutionContext execution;
};

struct DenseEmissionRequest {
    const CudaRiFrame& frame;
    CudaExecutionContext execution;
    int record_capacity;
    bool use_keep_flags;
};

struct DenseEmissionResult {
    int valid_record_count{};
    int overflow{};
};

struct DenseCompactionRequest {
    CudaExecutionContext execution;
    int valid_record_count;
};

void mark_dense_candidate_faces(DenseCandidateRequest request);
void filter_isolated_component_slots_cuda(DenseIsolationRequest request);
DenseEmissionResult emit_dense_candidate_records(DenseEmissionRequest request);
thrust::device_vector<Triangle>& compact_dense_face_records(DenseCompactionRequest request);

}  // namespace ri_meshing::cuda_detail
