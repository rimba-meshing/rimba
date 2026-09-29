#pragma once

#include "ri_meshing/boundary_graph.hpp"
#include "stitch_state.hpp"
#include "ri_meshing/config.hpp"
#include "ri_meshing/timing.hpp"

namespace ri_meshing::cuda_pipeline {

struct SpatialBoundaryFilterRequest {
    const std::vector<Vec3f>& vertices_global;
    const ri_meshing::BoundaryGraph::EdgeBlocks& old_boundary_edge_blocks;
    std::size_t old_boundary_edge_count{};
    const std::vector<int>& old_boundary_vertex_ids;
    const std::vector<std::array<int, 2>>& new_boundary_edges;
    int old_global_vertex_count{};
    const SparseMeshingConfig& config;
};

MeshScalar stitch_spatial_filter_radius(const SparseMeshingConfig& config);
void apply_active_boundary_changes(
    StitchSpatialBoundaryIndex& index,
    const std::vector<Vec3f>& vertices_global,
    const std::vector<int>& current_boundary_vertex_ids,
    const std::vector<int>& changed_vertex_ids,
    MeshScalar cell_size
);
void filter_old_boundary_edges_near_new(
    const SpatialBoundaryFilterRequest& request,
    StitchSpatialBoundaryIndex& spatial_index,
    FilteredOldBoundaryEdges& result,
    PipelineTiming* timing
);

}  // namespace ri_meshing::cuda_pipeline
