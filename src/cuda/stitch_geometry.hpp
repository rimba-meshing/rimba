#pragma once

#include "ri_meshing/timing.hpp"
#include "ri_meshing/types.hpp"

#include <vector>

namespace ri_meshing::cuda_detail {
struct CudaWorkspace;
}

namespace ri_meshing::cuda_pipeline {

Mat4f inverse_rigid_cpu(Mat4f pose);
Vec3f transform_point_cpu(Vec3f point, const Mat4f& transform);
void sync_new_host_vertices(
    cuda_detail::CudaWorkspace& workspace,
    std::vector<Vec3f>& host_vertices_global,
    int old_vertex_count,
    int new_vertex_count,
    PipelineTiming* timing
);
void accumulate_host_face_normal_sums(
    const std::vector<Vec3f>& vertices_global,
    const std::vector<Triangle>& faces,
    std::vector<Vec3f>& normal_sums_global,
    PipelineTiming* timing
);

}  // namespace ri_meshing::cuda_pipeline
