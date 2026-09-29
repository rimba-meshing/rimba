#include "ri_meshing/config.hpp"

#include <cassert>
#include <stdexcept>

int main()
{
    ri_meshing::SparseMeshingConfig config{};
    ri_meshing::validate_config(config);

    config.ri_height = 0;
    try {
        ri_meshing::validate_config(config);
        assert(false && "invalid config should throw");
    } catch (const std::invalid_argument&) {
    }

    config = ri_meshing::SparseMeshingConfig{};
    config.vertical_fov.min_rad = config.vertical_fov.max_rad;
    try {
        ri_meshing::validate_config(config);
        assert(false && "invalid fov should throw");
    } catch (const std::invalid_argument&) {
    }

    config = ri_meshing::SparseMeshingConfig{};
    config.dense_isolated_component_max_faces = -1;
    try {
        ri_meshing::validate_config(config);
        assert(false && "invalid dense isolated component limit should throw");
    } catch (const std::invalid_argument&) {
    }

    config = ri_meshing::SparseMeshingConfig{};
    config.stitch_connection_frontier_expand_radius = -1;
    try {
        ri_meshing::validate_config(config);
        assert(false && "invalid frontier expansion radius should throw");
    } catch (const std::invalid_argument&) {
    }

    config = ri_meshing::SparseMeshingConfig{};
    config.hole_fill_max_triangle_dist = 0.0;
    try {
        ri_meshing::validate_config(config);
        assert(false && "invalid hole-fill triangle distance should throw");
    } catch (const std::invalid_argument&) {
    }

    config = ri_meshing::SparseMeshingConfig{};
    config.accumulated_depth_cull_radius = 0.0;
    try {
        ri_meshing::validate_config(config);
        assert(false && "invalid accumulated-depth culling radius should throw");
    } catch (const std::invalid_argument&) {
    }

    return 0;
}
