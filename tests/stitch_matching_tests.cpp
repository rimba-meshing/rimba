#include "stitch_core_components.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace {

using ri_meshing::BoundaryPath;
using ri_meshing::SparseMeshingConfig;
using ri_meshing::StitchConnectionMask;
using ri_meshing::Vec3f;
using ri_meshing::stitch_detail::BoundaryMatchingRequest;
using ri_meshing::stitch_detail::BoundaryMatchingResult;
using ri_meshing::stitch_detail::StitchConnectionOracle;
using ri_meshing::stitch_detail::StitchConnectionOracleRequest;
using ri_meshing::stitch_detail::match_boundaries;

StitchConnectionMask connection_mask()
{
    StitchConnectionMask mask;
    mask.height = 64;
    mask.width = 64;
    mask.pixels.assign(static_cast<std::size_t>(mask.height * mask.width), 0);
    mask.pixels[32 * mask.width + 32] = 1;
    return mask;
}

BoundaryMatchingResult match_internal_gap_case(bool gap_at_endpoint)
{
    const std::vector<Vec3f> vertices{
        {1.0F, 0.0F, 0.0F},
        {1.6F, 0.0F, 0.0F},
        {1.6F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F},
    };
    const std::vector<BoundaryPath> old_paths{
        gap_at_endpoint ? BoundaryPath{{0, 1, 2, 3}, false}
                         : BoundaryPath{{3, 1, 2, 0}, false}
    };
    const std::vector<BoundaryPath> new_paths{{{4, 5, 6}, false}};
    const std::vector<std::array<int, 2>> old_edges{{0, 1}, {1, 2}, {2, 3}};
    const std::vector<std::array<int, 2>> new_edges{{4, 5}, {5, 6}};
    const std::vector<Vec3f> normals(vertices.size());
    SparseMeshingConfig config;
    config.ri_height = 64;
    config.ri_width = 64;
    config.boundary_label_max_euclidean_dist = 0.25;
    const StitchConnectionMask mask = connection_mask();
    StitchConnectionOracle oracle(
        StitchConnectionOracleRequest{mask, config, vertices, old_paths, new_paths}
    );
    return match_boundaries(
        BoundaryMatchingRequest{
            vertices,
            old_edges,
            new_edges,
            normals,
            old_paths,
            new_paths,
            4,
            true,
            &oracle,
        },
        config,
        nullptr,
        nullptr
    );
}

BoundaryMatchingResult match_wrapped_segment_case()
{
    const std::vector<Vec3f> vertices{
        {1.0F, 0.20F, 0.0F},
        {1.0F, 0.0F, 0.0F},
        {10.0F, 0.0F, 0.0F},
        {10.0F, 0.0F, 0.0F},
        {10.0F, 0.0F, 0.0F},
        {1.0F, 0.60F, 0.0F},
        {1.0F, 0.40F, 0.0F},
        {1.0F, 0.0F, 0.0F},
        {1.0F, 0.20F, 0.0F},
        {1.0F, 0.40F, 0.0F},
        {1.0F, 0.60F, 0.0F},
    };
    const std::vector<BoundaryPath> old_paths{
        BoundaryPath{{0, 1, 2, 3, 4, 5, 6}, true}
    };
    const std::vector<BoundaryPath> new_paths{
        BoundaryPath{{7, 8, 9, 10}, true}
    };
    const std::vector<std::array<int, 2>> old_edges{
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 0}
    };
    const std::vector<std::array<int, 2>> new_edges{
        {7, 8}, {8, 9}, {9, 10}, {10, 7}
    };
    const std::vector<Vec3f> normals(vertices.size());
    SparseMeshingConfig config;
    config.ri_height = 64;
    config.ri_width = 64;
    config.boundary_label_max_euclidean_dist = 0.25;
    StitchConnectionMask mask = connection_mask();
    for (int col = 32; col <= 50; ++col) {
        mask.pixels[32 * mask.width + col] = 1;
    }
    StitchConnectionOracle oracle(
        StitchConnectionOracleRequest{mask, config, vertices, old_paths, new_paths}
    );
    return match_boundaries(
        BoundaryMatchingRequest{
            vertices,
            old_edges,
            new_edges,
            normals,
            old_paths,
            new_paths,
            7,
            true,
            &oracle,
        },
        config,
        nullptr,
        nullptr
    );
}

void test_internal_two_vertex_gap_is_repaired()
{
    const BoundaryMatchingResult result = match_internal_gap_case(false);
    assert(result.old_segments_by_cluster.size() == 1);
    assert(result.old_segments_by_cluster[0].size() == 1);
    const auto& segment = result.old_segments_by_cluster[0][0];
    assert((segment.ids == std::vector<int>{3, 1, 2, 0}));
    assert(segment.labels.size() == 4);
    assert(segment.labels[1] == segment.labels[0]);
    assert(segment.labels[2] == segment.labels[0]);
}

void test_endpoint_two_vertex_gap_is_not_repaired()
{
    const BoundaryMatchingResult result = match_internal_gap_case(true);
    assert(result.old_segments_by_cluster.size() == 1);
    assert(result.old_segments_by_cluster[0].size() == 1);
    const auto& segment = result.old_segments_by_cluster[0][0];
    assert((segment.ids == std::vector<int>{2, 3}));
}

void test_closed_path_segments_merge_across_wrap()
{
    const BoundaryMatchingResult result = match_wrapped_segment_case();
    assert(result.old_segments_by_cluster.size() == 1);
    assert(result.old_segments_by_cluster[0].size() == 1);
    const auto& segment = result.old_segments_by_cluster[0][0];
    assert(segment.source_path == 0);
    assert(segment.ids.size() == 4);
    assert(segment.ids[1] == 0);
    assert(segment.ids[2] == 6);
}

}  // namespace

int main()
{
    test_closed_path_segments_merge_across_wrap();
    test_internal_two_vertex_gap_is_repaired();
    test_endpoint_two_vertex_gap_is_not_repaired();
    return 0;
}
