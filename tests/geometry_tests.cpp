#include "ri_meshing/face_orientation.hpp"
#include "ri_meshing/boundary_graph.hpp"
#include "ri_meshing/boundary_paths.hpp"
#include "ri_meshing/geometry.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

void test_projection()
{
    ri_meshing::SparseMeshingConfig config{};
    const auto pixel = ri_meshing::project_to_ri({1.0, 0.0, 0.0}, config);
    assert(pixel.has_value());
    assert(pixel->row == 1024);
    assert(pixel->col == 1024);
}

void test_triangle_metrics()
{
    ri_meshing::SparseMeshingConfig config{};
    config.max_triangle_dist = 2.0;
    const auto metrics = ri_meshing::triangle_metrics(
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    );
    assert(std::abs(metrics.twice_area - 1.0) < 1e-12);
    assert(std::abs(metrics.longest_edge_sq - 2.0) < 1e-12);
    assert(ri_meshing::triangle_passes_filters(
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        config
    ));
}

void test_face_orientation()
{
    const std::vector<ri_meshing::Vec3f> vertices{
        {1.0F, 0.0F, 0.0F},
        {1.0F, 1.0F, 0.0F},
        {1.0F, 0.0F, 1.0F},
    };
    ri_meshing::Triangle face{0, 1, 2};
    assert(ri_meshing::sensor_orientation_dot(
               vertices[0], vertices[1], vertices[2]) > 0.0);
    assert(ri_meshing::orient_face_toward_sensor(face, vertices));
    assert(face.v1 == 2 && face.v2 == 1);
    assert(ri_meshing::sensor_orientation_dot(
               vertices[face.v0], vertices[face.v1], vertices[face.v2]) < 0.0);
}

void test_oriented_boundary_paths()
{
    const std::vector<ri_meshing::BoundaryEdge> edges{
        {2, 3}, {0, 1}, {3, 0}, {1, 2},
    };
    const auto paths = ri_meshing::extract_oriented_boundary_paths(edges);
    assert(paths.size() == 1);
    assert(paths.front().closed);
    assert((paths.front().ids == std::vector<int>{0, 1, 2, 3}));

    const std::vector<ri_meshing::BoundaryEdge> reversed{
        {1, 0}, {2, 1}, {3, 2}, {0, 3},
    };
    const auto reversed_paths = ri_meshing::extract_oriented_boundary_paths(reversed);
    assert(reversed_paths.size() == 1);
    assert((reversed_paths.front().ids == std::vector<int>{0, 3, 2, 1}));

    const std::vector<ri_meshing::BoundaryEdge> one_reversed_edge{
        {2, 3}, {0, 1}, {3, 0}, {2, 1},
    };
    const auto repaired = ri_meshing::extract_oriented_boundary_paths(one_reversed_edge);
    assert(repaired.size() == 1);
    assert((repaired.front().ids == std::vector<int>{0, 1, 2, 3}));

    const std::vector<ri_meshing::BoundaryEdge> ambiguous{
        {0, 1}, {2, 1}, {2, 3}, {0, 3},
    };
    const auto ambiguous_paths = ri_meshing::extract_oriented_boundary_paths(ambiguous);
    assert(ambiguous_paths.empty());
}

void test_local_boundary_collection()
{
    ri_meshing::BoundaryGraph graph;
    graph.toggle_boundary_edges({
        {2, 3}, {0, 1}, {3, 0}, {1, 2},
        {10, 11}, {11, 12}, {12, 13}, {13, 10},
    });

    const auto initial_paths = graph.oriented_boundary_paths();
    assert(initial_paths.size() == 2);

    graph.toggle_boundary_edges({{0, 1}});
    const auto updated_paths = graph.oriented_boundary_paths();
    assert(updated_paths.size() == 1);
}

void test_boundary_toggle_merge()
{
    ri_meshing::BoundaryGraph graph;
    graph.toggle_boundary_edges({
        {10, 11}, {20, 21}, {30, 31},
    });
    graph.toggle_boundary_edges({
        {10, 11}, {12, 13}, {40, 41},
    });
    assert((graph.boundary_edges() == std::vector<ri_meshing::BoundaryGraph::Edge>{
        {12, 13}, {20, 21}, {30, 31}, {40, 41},
    }));

    graph.toggle_boundary_edges({
        {12, 13}, {25, 26}, {30, 31},
    });
    assert((graph.boundary_edges() == std::vector<ri_meshing::BoundaryGraph::Edge>{
        {20, 21}, {25, 26}, {40, 41},
    }));
    assert((graph.boundary_vertex_ids() == std::vector<int>{20, 21, 25, 26, 40, 41}));
}

void test_blocked_boundary_storage()
{
    ri_meshing::BoundaryGraph graph;
    std::vector<ri_meshing::BoundaryGraph::Edge> initial;
    initial.reserve(10000);
    for (int index = 0; index < 10000; ++index) {
        initial.push_back({index * 2, index * 2 + 1});
    }
    graph.toggle_boundary_edges(initial);
    assert(graph.boundary_edge_count() == initial.size());
    assert(graph.boundary_edge_blocks().size() > 1);
    assert(graph.boundary_edges() == initial);

    graph.toggle_boundary_edges({{0, 1}, {8192, 8193}, {20001, 20002}});
    initial.erase(initial.begin());
    initial.erase(initial.begin() + 4095);
    initial.push_back({20001, 20002});
    std::sort(initial.begin(), initial.end(), [](const auto first, const auto second) {
        return first < second;
    });
    assert(graph.boundary_edges() == initial);
    for (const auto& block : graph.boundary_edge_blocks()) {
        assert(!block.empty());
    }
}

void test_seeded_hole_paths()
{
    ri_meshing::BoundaryGraph graph;
    graph.toggle_boundary_edges({
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {10, 11}, {11, 12}, {12, 13}, {13, 10},
    });

    ri_meshing::BoundaryPathExtractionStats stats;
    const auto seeded_paths = graph.oriented_boundary_paths_for_holes(
        {{{0, 1}, {1, 2}, {2, 3}, {3, 0}}}, 40, &stats
    );
    assert(seeded_paths.size() == 1);
    assert(seeded_paths.front().closed);
    assert((seeded_paths.front().ids == std::vector<int>{0, 1, 2, 3}));
    assert(stats.dirty_seed_vertices == 4);
    assert(stats.seeded_walk_vertices == 4);
    assert(stats.seeded_walk_steps == 4);

    const auto bounded_paths = graph.oriented_boundary_paths_for_holes(
        {{{0, 1}}}, 3, nullptr
    );
    assert(bounded_paths.empty());

    const auto all_paths = graph.oriented_boundary_paths();
    assert(all_paths.size() == 2);

    ri_meshing::BoundaryGraph branched_graph;
    branched_graph.toggle_boundary_edges({
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}
    });
    const auto branched_paths = branched_graph.oriented_boundary_paths_for_holes(
        {{{0, 1}}}, 40, nullptr
    );
    assert(branched_paths.empty());
}

}  // namespace

int main()
{
    test_projection();
    test_triangle_metrics();
    test_face_orientation();
    test_oriented_boundary_paths();
    test_local_boundary_collection();
    test_boundary_toggle_merge();
    test_blocked_boundary_storage();
    test_seeded_hole_paths();
    return 0;
}
