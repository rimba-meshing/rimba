#pragma once

#include "stitch_core_components.hpp"

#include <vector>

namespace ri_meshing::stitch_detail {

BoundaryIncidenceData build_boundary_incidence(
    const std::vector<BoundaryPath>& paths,
    const std::vector<Edge>& boundary_edges,
    const std::vector<Vec3f>& vertices,
    const std::vector<Vec3f>& vertex_normals
);

void annotate_open_side_components(
    BoundaryIncidenceData& incidence,
    const StitchConnectionOracle& connection_oracle,
    bool source_is_old
);

void annotate_open_side_component_pair(
    BoundaryIncidenceData& old_incidence,
    BoundaryIncidenceData& new_incidence,
    const StitchConnectionOracle& connection_oracle
);

bool has_open_side_match(
    int old_vertex,
    int new_vertex,
    const std::vector<Vec3f>& vertices,
    const BoundaryIncidenceData& old_incidence,
    const BoundaryIncidenceData& new_incidence
);

}  // namespace ri_meshing::stitch_detail
