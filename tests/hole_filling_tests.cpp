#include "ri_meshing/hole_filling.hpp"

#include <cassert>
#include <vector>

namespace {

ri_meshing::Mat4f identity()
{
    return ri_meshing::Mat4f{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
}

ri_meshing::StitchConnectionMask seam_mask(const ri_meshing::SparseMeshingConfig& config)
{
    ri_meshing::StitchConnectionMask mask;
    mask.height = static_cast<int>(config.ri_height);
    mask.width = static_cast<int>(config.ri_width);
    mask.pixels.assign(config.ri_height * config.ri_width, 0);
    mask.pixels[1024 * mask.width + 1024] = 1;
    return mask;
}

std::vector<ri_meshing::Vec3f> square_vertices()
{
    return {
        {5.0F, -0.05F, -0.05F},
        {5.0F, 0.05F, -0.05F},
        {5.0F, 0.05F, 0.05F},
        {5.0F, -0.05F, 0.05F},
    };
}

void test_mixed_seam_loop_is_filled()
{
    ri_meshing::SparseMeshingConfig config;
    const auto vertices = square_vertices();
    const auto mask = seam_mask(config);
    const std::vector<ri_meshing::BoundaryPath> paths{{{0, 1, 2, 3}, true}};
    const ri_meshing::HoleFillResult result = ri_meshing::fill_local_seam_holes(
        ri_meshing::HoleFillRequest{vertices, paths, &mask, config, 2, identity()}
    );

    assert(result.candidate_loops == 1);
    assert(result.accepted_loops == 1);
    assert(result.vertices_global.size() == 1);
    assert(result.faces.size() == 4);
    for (const ri_meshing::Triangle face : result.faces) {
        assert(face.v0 < 4 || face.v0 == 4);
        assert(face.v1 < 4 || face.v1 == 4);
        assert(face.v2 < 4 || face.v2 == 4);
    }
}

void test_old_only_loop_is_not_filled()
{
    ri_meshing::SparseMeshingConfig config;
    const auto vertices = square_vertices();
    const auto mask = seam_mask(config);
    const std::vector<ri_meshing::BoundaryPath> paths{{{0, 1, 2, 3}, true}};
    const ri_meshing::HoleFillResult result = ri_meshing::fill_local_seam_holes(
        ri_meshing::HoleFillRequest{vertices, paths, &mask, config, 4, identity()}
    );

    assert(result.candidate_loops == 0);
    assert(result.vertices_global.empty());
    assert(result.faces.empty());
}

void test_mixed_seam_loop_accepts_adjacent_support()
{
    ri_meshing::SparseMeshingConfig config;
    const auto vertices = square_vertices();
    auto mask = seam_mask(config);
    mask.pixels.assign(config.ri_height * config.ri_width, 0);
    mask.pixels[1024 * mask.width + 1020] = 1;
    const std::vector<ri_meshing::BoundaryPath> paths{{{0, 1, 2, 3}, true}};
    const ri_meshing::HoleFillResult result = ri_meshing::fill_local_seam_holes(
        ri_meshing::HoleFillRequest{vertices, paths, &mask, config, 2, identity()}
    );

    assert(result.candidate_loops == 1);
    assert(result.accepted_loops == 1);
    assert(result.faces.size() == 4);
}

void test_clockwise_island_loop_is_not_filled()
{
    ri_meshing::SparseMeshingConfig config;
    const auto vertices = square_vertices();
    const auto mask = seam_mask(config);
    const std::vector<ri_meshing::BoundaryPath> paths{{{0, 3, 2, 1}, true}};
    const ri_meshing::HoleFillResult result = ri_meshing::fill_local_seam_holes(
        ri_meshing::HoleFillRequest{vertices, paths, &mask, config, 2, identity()}
    );

    assert(result.candidate_loops == 0);
    assert(result.accepted_loops == 0);
    assert(result.vertices_global.empty());
    assert(result.faces.empty());
}

void test_hole_fill_distance_is_independent_from_stitch_distance()
{
    ri_meshing::SparseMeshingConfig config;
    config.stitch_max_triangle_dist = 0.05;
    config.hole_fill_max_triangle_dist = 0.20;
    const auto vertices = square_vertices();
    const auto mask = seam_mask(config);
    const std::vector<ri_meshing::BoundaryPath> paths{{{0, 1, 2, 3}, true}};
    const ri_meshing::HoleFillResult result = ri_meshing::fill_local_seam_holes(
        ri_meshing::HoleFillRequest{vertices, paths, &mask, config, 2, identity()}
    );

    assert(result.candidate_loops == 1);
    assert(result.accepted_loops == 1);
    assert(result.faces.size() == 4);
}

}  // namespace

int main()
{
    test_mixed_seam_loop_is_filled();
    test_old_only_loop_is_not_filled();
    test_mixed_seam_loop_accepts_adjacent_support();
    test_clockwise_island_loop_is_not_filled();
    test_hole_fill_distance_is_independent_from_stitch_distance();
    return 0;
}
