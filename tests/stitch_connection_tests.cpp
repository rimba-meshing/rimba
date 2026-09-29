#include "stitch_connection.hpp"

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <utility>
#include <vector>

namespace {

using ri_meshing::BoundaryPath;
using ri_meshing::SparseMeshingConfig;
using ri_meshing::StitchConnectionMask;
using ri_meshing::Vec3f;
using ri_meshing::stitch_detail::StitchConnectionOracle;
using ri_meshing::stitch_detail::StitchConnectionOracleRequest;

Vec3f spherical(double azimuth, double elevation)
{
    const double cos_elevation = std::cos(elevation);
    return Vec3f{
        static_cast<float>(cos_elevation * std::cos(azimuth)),
        static_cast<float>(cos_elevation * std::sin(azimuth)),
        static_cast<float>(std::sin(elevation)),
    };
}

Vec3f ri_pixel(int row, int col, int height, int width)
{
    const double elevation = -0.7853981633974483
        + static_cast<double>(row) / static_cast<double>(height) * 1.5707963267948966;
    const double azimuth = -std::acos(-1.0)
        + static_cast<double>(col) / static_cast<double>(width) * 2.0 * std::acos(-1.0);
    return spherical(azimuth, elevation);
}

StitchConnectionMask mask_with_pixels(std::initializer_list<std::pair<int, int>> pixels)
{
    StitchConnectionMask mask;
    mask.height = 9;
    mask.width = 16;
    mask.pixels.assign(static_cast<std::size_t>(mask.height * mask.width), 0);
    for (const auto [row, col] : pixels) {
        mask.pixels[static_cast<std::size_t>(row * mask.width + col)] = 1;
    }
    return mask;
}

StitchConnectionMask mask_with_old_linear_support()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}});
    mask.old_support.assign(static_cast<std::size_t>(mask.height * mask.width), 0);
    for (const auto [row, col] : {
        std::pair<int, int>{7, 3}, {6, 3}, {6, 4}, {5, 4}, {5, 5}
    }) {
        mask.old_support[static_cast<std::size_t>(row * mask.width + col)] = 1;
    }
    return mask;
}

void test_horizontal_wrap_keeps_component_connected()
{
    const StitchConnectionMask mask = mask_with_pixels({{4, 0}, {4, 15}});
    const std::vector<Vec3f> vertices{
        spherical(-3.13, 0.0), spherical(3.13, 0.0),
        spherical(-3.13, 0.15), spherical(3.13, 0.15),
    };
    const std::vector<BoundaryPath> old_paths{{{0, 1}, false}};
    const std::vector<BoundaryPath> new_paths{{{2, 3}, false}};
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, old_paths, new_paths,
    });

    assert(oracle.pair_allowed(0, 2));
}

void test_positive_pi_uses_last_pixel_before_segment_wrap()
{
    const StitchConnectionMask mask = mask_with_pixels({
        {4, 15}, {5, 15}, {6, 15}, {7, 15},
    });
    const double pi = std::acos(-1.0);
    const std::vector<Vec3f> vertices{
        spherical(pi, 0.0), spherical(pi, 0.1),
        spherical(pi, 0.2), spherical(pi, 0.3),
    };
    const std::vector<BoundaryPath> old_paths{{{0, 1}, false}};
    const std::vector<BoundaryPath> new_paths{{{2, 3}, false}};
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, old_paths, new_paths,
    });

    assert(oracle.pair_allowed(0, 2));
}

void test_different_components_do_not_cross()
{
    const StitchConnectionMask mask = mask_with_pixels({{4, 2}, {4, 13}});
    const std::vector<Vec3f> vertices{
        spherical(-2.35, 0.0), spherical(-2.20, 0.0),
        spherical(2.35, 0.15), spherical(2.20, 0.15),
    };
    const std::vector<BoundaryPath> old_paths{{{0, 1}, false}};
    const std::vector<BoundaryPath> new_paths{{{2, 3}, false}};
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, old_paths, new_paths,
    });

    assert(!oracle.pair_allowed(0, 2));
}

void test_diagonal_red_pixels_share_component()
{
    const StitchConnectionMask mask = mask_with_pixels({{4, 2}, {5, 3}});
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 2, mask.height, mask.width),
        ri_pixel(5, 3, mask.height, mask.width),
    };
    const std::vector<BoundaryPath> old_paths{{{0}, false}};
    const std::vector<BoundaryPath> new_paths{{{1}, false}};
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, old_paths, new_paths,
    });

    assert(oracle.pair_allowed(0, 1));
}

void test_inherited_component_support_allows_single_gap_vertex()
{
    const StitchConnectionMask mask = mask_with_pixels({{4, 2}});
    const std::vector<Vec3f> vertices{
        spherical(-2.35, 0.0), spherical(-2.20, 0.0),
        spherical(-2.35, 0.15), spherical(-2.20, 0.15),
        spherical(-2.35, 0.0),
    };
    const std::vector<BoundaryPath> old_paths{{{0, 1}, false}};
    const std::vector<BoundaryPath> new_paths{{{2, 3}, false}};
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, old_paths, new_paths,
    });

    assert(!oracle.pair_allowed(4, 2));
    oracle.inherit_component_support(4, 0);
    assert(oracle.pair_allowed(4, 2));
}

void test_bounded_linear_old_support_handoff()
{
    const StitchConnectionMask mask = mask_with_old_linear_support();
    const std::vector<Vec3f> vertices{
        ri_pixel(7, 3, mask.height, mask.width),
        ri_pixel(7, 4, mask.height, mask.width),
        ri_pixel(4, 5, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0, 1}, false}}, {{{2}, false}},
    });
    assert(oracle.pair_allowed(0, 2));
}

void test_bounded_linear_handoff_rejects_non_old_gap()
{
    StitchConnectionMask mask = mask_with_old_linear_support();
    mask.old_support[5 * mask.width + 4] = 0;
    const std::vector<Vec3f> vertices{
        ri_pixel(7, 3, mask.height, mask.width),
        ri_pixel(7, 4, mask.height, mask.width),
        ri_pixel(4, 5, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0, 1}, false}}, {{{2}, false}},
    });
    assert(!oracle.pair_allowed(0, 2));
}

void test_long_red_component_handoff_is_rejected()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}, {5, 5}, {6, 5}, {7, 5}, {8, 5}});
    mask.old_support.assign(static_cast<std::size_t>(mask.height * mask.width), 1);
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 5, mask.height, mask.width),
        ri_pixel(8, 5, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    config.stitch_connection_max_red_steps = 3;
    StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{1}, false}},
    });

    assert(!oracle.pair_allowed(0, 1));
    assert(oracle.pair_allowed_for_geometry(0, 1));
}

void test_nearest_empty_pixel_blocks_red_support()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}});
    mask.old_support.assign(static_cast<std::size_t>(mask.height * mask.width), 0);
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 6, mask.height, mask.width),
        ri_pixel(4, 5, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{1}, false}},
    });

    assert(!oracle.pair_allowed(0, 1));
}

void test_red_support_wins_empty_pixel_tie()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}});
    mask.old_support.assign(static_cast<std::size_t>(mask.height * mask.width), 0);
    mask.old_support[4 * mask.width + 6] = 1;
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 6, mask.height, mask.width),
        ri_pixel(4, 5, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{1}, false}},
    });

    assert(oracle.pair_allowed(0, 1));
}

void test_new_pixel_is_not_treated_as_empty()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}});
    mask.old_support.assign(static_cast<std::size_t>(mask.height * mask.width), 0);
    mask.new_support.assign(static_cast<std::size_t>(mask.height * mask.width), 0);
    mask.new_support[4 * mask.width + 6] = 1;
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 6, mask.height, mask.width),
        ri_pixel(4, 5, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{1}, false}},
    });

    assert(oracle.pair_allowed(0, 1));
}

void test_open_side_walks_own_mesh_to_red_component()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}});
    const std::size_t pixel_count = static_cast<std::size_t>(mask.height * mask.width);
    mask.old_support.assign(pixel_count, 0);
    mask.new_support.assign(pixel_count, 0);
    mask.old_support[4 * mask.width + 3] = 1;
    mask.old_support[4 * mask.width + 4] = 1;
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 3, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{0}, false}},
    });

    assert(oracle.component_for_open_side(0, 1.0, 0.0, true) == 0);
}

void test_open_side_accepts_red_component_at_origin()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 3}});
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 3, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{0}, false}},
    });

    assert(oracle.component_for_open_side(0, 1.0, 0.0, true) == 0);
    assert(oracle.component_for_open_side(0, -1.0, 0.0, false) == 0);
}

void test_open_side_blocks_opposite_mesh_after_own_support()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}});
    const std::size_t pixel_count = static_cast<std::size_t>(mask.height * mask.width);
    mask.old_support.assign(pixel_count, 0);
    mask.new_support.assign(pixel_count, 0);
    mask.old_support[4 * mask.width + 3] = 1;
    mask.old_support[4 * mask.width + 4] = 1;
    mask.new_support[4 * mask.width + 4] = 1;
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 3, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{0}, false}},
    });

    assert(oracle.component_for_open_side(0, 1.0, 0.0, true) == -1);
}

void test_open_side_new_mesh_uses_same_walk_rule()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 5}});
    const std::size_t pixel_count = static_cast<std::size_t>(mask.height * mask.width);
    mask.old_support.assign(pixel_count, 0);
    mask.new_support.assign(pixel_count, 0);
    mask.new_support[4 * mask.width + 3] = 1;
    mask.new_support[4 * mask.width + 4] = 1;
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 3, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{0}, false}},
    });

    assert(oracle.component_for_open_side(0, 1.0, 0.0, false) == 0);
}

void test_vertex_keeps_only_nearest_red_component()
{
    StitchConnectionMask mask = mask_with_pixels({{4, 4}, {4, 8}});
    mask.old_support.assign(static_cast<std::size_t>(mask.height * mask.width), 1);
    const std::vector<Vec3f> vertices{
        ri_pixel(4, 6, mask.height, mask.width),
        ri_pixel(4, 4, mask.height, mask.width),
        ri_pixel(4, 8, mask.height, mask.width),
    };
    SparseMeshingConfig config;
    config.ri_height = mask.height;
    config.ri_width = mask.width;
    config.stitch_connection_max_ri_radius = 3;
    const StitchConnectionOracle oracle(StitchConnectionOracleRequest{
        mask, config, vertices, {{{0}, false}}, {{{1}, false}},
    });

    assert(oracle.pair_allowed(0, 1));
    assert(!oracle.pair_allowed(0, 2));
}

}  // namespace

int main()
{
    test_horizontal_wrap_keeps_component_connected();
    test_positive_pi_uses_last_pixel_before_segment_wrap();
    test_different_components_do_not_cross();
    test_diagonal_red_pixels_share_component();
    test_inherited_component_support_allows_single_gap_vertex();
    test_bounded_linear_old_support_handoff();
    test_bounded_linear_handoff_rejects_non_old_gap();
    test_long_red_component_handoff_is_rejected();
    test_nearest_empty_pixel_blocks_red_support();
    test_red_support_wins_empty_pixel_tie();
    test_new_pixel_is_not_treated_as_empty();
    test_open_side_walks_own_mesh_to_red_component();
    test_open_side_accepts_red_component_at_origin();
    test_open_side_blocks_opposite_mesh_after_own_support();
    test_open_side_new_mesh_uses_same_walk_rule();
    test_vertex_keeps_only_nearest_red_component();
    return 0;
}
