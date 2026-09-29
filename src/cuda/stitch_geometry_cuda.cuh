#pragma once

#include "cuda_workflow.hpp"
#include "stitch_geometry.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

namespace ri_meshing::cuda_pipeline {

struct DeviceNormalAccumulationRequest {
    cuda_detail::CudaWorkspace& workspace;
    const thrust::device_vector<Triangle>& faces;
    int first_new_vertex{};
    int new_vertex_count{};
    std::vector<Vec3f>& normal_sums_global;
    std::vector<Vec3f>& host_current_vertex_normal_sums;
    PipelineTiming* timing{};
    cudaStream_t stream{};
};

void launch_device_face_normal_sums(DeviceNormalAccumulationRequest request);
void materialize_device_face_normal_sums(DeviceNormalAccumulationRequest request);

void accumulate_device_face_normal_sums(DeviceNormalAccumulationRequest request);

}  // namespace ri_meshing::cuda_pipeline
