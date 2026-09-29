#include "ri_meshing/config.hpp"

#include <stdexcept>

namespace ri_meshing {

void validate_config(const SparseMeshingConfig& config)
{
    if (config.ri_height == 0 || config.ri_width == 0) {
        throw std::invalid_argument("ri dimensions must be positive");
    }
    if (config.vertical_fov.min_rad >= config.vertical_fov.max_rad) {
        throw std::invalid_argument("vertical_fov must be ordered");
    }
    if (config.max_triangle_dist <= 0.0) {
        throw std::invalid_argument("triangle distance must be positive");
    }
    if (config.accumulated_depth_cull_radius <= 0.0) {
        throw std::invalid_argument("accumulated-depth culling radius must be positive");
    }
    if (config.boundary_label_max_euclidean_dist <= 0.0) {
        throw std::invalid_argument("boundary label distance must be positive");
    }
    if (config.stitch_max_triangle_dist <= 0.0) {
        throw std::invalid_argument("stitch triangle distance must be positive");
    }
    if (config.hole_fill_max_triangle_dist <= 0.0) {
        throw std::invalid_argument("hole-fill triangle distance must be positive");
    }
    if (config.stitch_connection_frontier_expand_radius < 0) {
        throw std::invalid_argument(
            "stitch connection frontier expansion radius must be non-negative"
        );
    }
    if (config.stitch_connection_max_ri_radius < 0) {
        throw std::invalid_argument("stitch connection RI radius must be non-negative");
    }
    if (config.stitch_connection_max_red_steps < 0) {
        throw std::invalid_argument("stitch connection red steps must be non-negative");
    }
    if (config.overlap_depth_abs_tol < 0.0 || config.overlap_depth_rel_tol < 0.0) {
        throw std::invalid_argument("overlap depth tolerances must be non-negative");
    }
    if (config.overlap_min_pixels <= 0) {
        throw std::invalid_argument("overlap_min_pixels must be positive");
    }
    if (config.dense_isolated_component_max_faces < 0) {
        throw std::invalid_argument("dense isolated component limit must be non-negative");
    }
    if (config.overlap_neighbor_euclidean_tol < 0.0) {
        throw std::invalid_argument("overlap neighbor tolerance must be non-negative");
    }
    if (config.stitch_retry_max_debt < 0) {
        throw std::invalid_argument("stitch iteration limits must be non-negative");
    }
    if (config.min_triangle_area < 0.0 || config.min_triangle_quality < 0.0) {
        throw std::invalid_argument("triangle quality thresholds must be non-negative");
    }
    if (config.sliver_repair_min_area < 0.0 || config.sliver_repair_min_quality < 0.0) {
        throw std::invalid_argument("sliver repair thresholds must be non-negative");
    }
    if (config.max_hole_edges <= 0 || config.max_hole_euclidean <= 0.0) {
        throw std::invalid_argument("hole-filling limits must be positive");
    }
    if (config.max_hole_ri_area_pixels <= 0.0) {
        throw std::invalid_argument("hole-fill RI area limit must be positive");
    }
    if (config.filled_pixel_macro_max_error < 0.0) {
        throw std::invalid_argument("filled-pixel macro error must be non-negative");
    }
}

}  // namespace ri_meshing
