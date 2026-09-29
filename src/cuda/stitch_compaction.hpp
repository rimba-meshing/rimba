#pragma once

#include "stitch_state.hpp"

namespace ri_meshing::cuda_pipeline {

struct BoundarySelection {
    const std::vector<std::array<int, 2>>& edges;
    const std::vector<int>& vertex_ids;
};

struct StitchCompactionRequest {
    const std::vector<Vec3f>& vertices_global;
    const std::vector<Vec3f>& normal_sums_global;
    BoundarySelection old_boundary;
    BoundarySelection new_boundary;
    int old_global_vertex_count{};
    Mat4f global_to_local{};
};

void compact_stitch_input(
    const StitchCompactionRequest& request,
    CompactStitchInput& out,
    std::vector<int>& global_to_compact,
    PipelineTiming* timing
);
void remap_faces_to_global(std::vector<Triangle>& faces, const std::vector<int>& compact_to_global);
std::vector<std::array<int, 2>> batch_boundary_edges_sorted(const std::vector<Triangle>& faces);

}  // namespace ri_meshing::cuda_pipeline
