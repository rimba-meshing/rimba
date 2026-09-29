#include "stitch_boundary_filters.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

using ri_meshing::BoundaryPath;
using ri_meshing::Vec3f;
using ri_meshing::stitch_detail::BoundaryIncidence;
using ri_meshing::stitch_detail::BoundaryIncidenceData;
using ri_meshing::stitch_detail::build_boundary_incidence;
using ri_meshing::stitch_detail::has_open_side_match;

Vec3f spherical(double azimuth, double elevation)
{
    const double cos_elevation = std::cos(elevation);
    return Vec3f{
        static_cast<float>(cos_elevation * std::cos(azimuth)),
        static_cast<float>(cos_elevation * std::sin(azimuth)),
        static_cast<float>(std::sin(elevation)),
    };
}

void test_angular_exterior_is_left_of_oriented_range_image_edge()
{
    const std::vector<Vec3f> vertices{
        spherical(-0.10, -0.10),
        spherical(0.10, -0.10),
        spherical(0.10, 0.10),
        spherical(-0.10, 0.10),
    };
    const std::vector<BoundaryPath> paths{{{0, 1, 2, 3}, true}};
    const std::vector<std::array<int, 2>> edges{
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
    };
    const BoundaryIncidenceData incidence = build_boundary_incidence(
        paths, edges, vertices, std::vector<Vec3f>(vertices.size())
    );
    const auto found = incidence.by_vertex.find(0);
    assert(found != incidence.by_vertex.end());

    bool found_rightward_edge = false;
    for (const BoundaryIncidence& item : found->second) {
        if (item.angular_tangent.x <= 0.9) {
            continue;
        }
        found_rightward_edge = true;
        assert(item.angular_open_dir.x < 0.1);
        assert(item.angular_open_dir.y > 0.9);
    }
    assert(found_rightward_edge);
}

void test_open_side_requires_a_shared_frontier_component()
{
    const std::vector<Vec3f> vertices(2);
    BoundaryIncidenceData old_incidence;
    BoundaryIncidenceData new_incidence;
    old_incidence.frontier_components_annotated = true;
    new_incidence.frontier_components_annotated = true;
    old_incidence.by_vertex[0].push_back(BoundaryIncidence{.frontier_component = 7});
    new_incidence.by_vertex[1].push_back(BoundaryIncidence{.frontier_component = 7});
    assert(has_open_side_match(0, 1, vertices, old_incidence, new_incidence));

    new_incidence.by_vertex[1][0].frontier_component = 8;
    assert(!has_open_side_match(0, 1, vertices, old_incidence, new_incidence));
}

}  // namespace

int main()
{
    test_angular_exterior_is_left_of_oriented_range_image_edge();
    test_open_side_requires_a_shared_frontier_component();
    return 0;
}
