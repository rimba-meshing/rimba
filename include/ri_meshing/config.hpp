#pragma once

#include <cstddef>
#include <string>

#include "ri_meshing/types.hpp"

namespace ri_meshing {

struct AngleRange {
    MeshScalar min_rad{static_cast<MeshScalar>(-0.7853981633974483)};
    MeshScalar max_rad{static_cast<MeshScalar>(0.7853981633974483)};
};

struct SparseMeshingConfig {
    std::string dataset_path{"~/Datasets/local/mai_city/bin/"};
    std::string sequence{"00"};
    bool use_custom_poses{false};

    std::size_t ri_height{2048};
    std::size_t ri_width{2048};
    AngleRange vertical_fov{};
    MeshScalar max_triangle_dist{0.30F};
    // Radius around the current sensor whose accumulated face chunks are
    // eligible to contribute to the temporary depth blocker.
    MeshScalar accumulated_depth_cull_radius{70.0F};
    // Number of 8-connected expansion layers added around base frontier pixels.
    int stitch_connection_frontier_expand_radius{2};
    // Maximum 8-connected RI steps for the bounded linear frontier handoff.
    int stitch_connection_max_ri_radius{6};
    // Maximum 8-connected red-component steps allowed for one label handoff.
    int stitch_connection_max_red_steps{3};
    int dense_isolated_component_max_faces{8};

    MeshScalar boundary_label_max_euclidean_dist{0.25F};
    MeshScalar overlap_depth_abs_tol{0.05F};
    MeshScalar overlap_depth_rel_tol{0.01F};
    int overlap_min_pixels{1};
    MeshScalar overlap_neighbor_euclidean_tol{0.05F};
    int stitch_retry_max_debt{10};
    MeshScalar stitch_max_triangle_dist{0.30F};
    // Local hole-fill patches may bridge a slightly longer boundary edge.
    MeshScalar hole_fill_max_triangle_dist{0.40F};
    MeshScalar sliver_repair_min_area{1e-8F};
    MeshScalar sliver_repair_min_quality{5e-5F};
    MeshScalar min_triangle_area{1e-5F};
    MeshScalar min_triangle_quality{1e-2F};
    int max_hole_edges{40};
    MeshScalar max_hole_euclidean{3.0F};
    // Maximum projected area of a local hole loop in RI pixel^2.
    MeshScalar max_hole_ri_area_pixels{50.0F};
    // Coarsening tries 16x16, 8x8, 4x4, 2x2 and finally 1x1 cells. The
    // coarse-face test deliberately does not use max_triangle_dist.
    MeshScalar filled_pixel_macro_max_error{0.08F};
};

void validate_config(const SparseMeshingConfig& config);

}  // namespace ri_meshing
