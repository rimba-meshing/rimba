#include "stitch_core_components.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <vector>

namespace {

bool has_face_with_vertices(
    const std::vector<ri_meshing::Triangle>& faces,
    std::array<int, 3> expected
)
{
    for (const ri_meshing::Triangle face : faces) {
        std::array<int, 3> actual{face.v0, face.v1, face.v2};
        std::sort(actual.begin(), actual.end());
        std::sort(expected.begin(), expected.end());
        if (actual == expected) {
            return true;
        }
    }
    return false;
}

bool has_face_with_vertices_pair(
    const std::vector<ri_meshing::Triangle>& faces,
    int first,
    int second
)
{
    for (const ri_meshing::Triangle face : faces) {
        const std::array<int, 3> actual{face.v0, face.v1, face.v2};
        const bool has_first = std::find(actual.begin(), actual.end(), first) != actual.end();
        const bool has_second = std::find(actual.begin(), actual.end(), second) != actual.end();
        if (has_first && has_second) {
            return true;
        }
    }
    return false;
}

std::vector<ri_meshing::Triangle> generate_square_candidates(bool closed)
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{
            {0, 1, 2, 3}, {0.0, 0.333333, 0.666667, 1.333333}, closed, -1, 0
        }
    };
    const std::vector<std::vector<OldLabelSegment>> grouped{old_segments};
    const std::vector<ri_meshing::Vec3f> vertices(8);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, grouped, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_partial_arc_candidates()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1, 2, 3}, {0.40, 0.45, 0.49, 0.50}, false}
    };
    const std::vector<ri_meshing::Vec3f> vertices(8);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_tied_partial_arc_candidates()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{3, 4, 5, 6, 7, 8, 9, 10}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1, 2}, {0.50, 0.50, 0.50}, false}
    };
    const std::vector<ri_meshing::Vec3f> vertices(11);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_tied_reversed_segment_candidates()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{3, 4, 5, 6, 7, 8, 9, 10}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{2, 1, 0}, {3.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0}, false}
    };
    std::vector<ri_meshing::Vec3f> vertices(11);
    vertices[0] = ri_meshing::Vec3f{0.0F, 1.0F, 0.0F};
    vertices[1] = ri_meshing::Vec3f{1.0F, 1.0F, 0.0F};
    vertices[2] = ri_meshing::Vec3f{2.0F, 1.0F, 0.0F};
    vertices[6] = ri_meshing::Vec3f{0.0F, 0.0F, 0.0F};
    vertices[7] = ri_meshing::Vec3f{1.0F, 0.0F, 0.0F};
    vertices[8] = ri_meshing::Vec3f{2.0F, 0.0F, 0.0F};
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_collapsed_closed_seam_candidate()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> old_paths{{{0, 1, 2, 3, 4, 5, 6}, true}};
    const std::vector<BoundaryPath> new_paths{{{6, 7, 8, 9, 10, 11, 12, 13, 14}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1}, {0.75, 0.65}, false, -1, 0},
        OldLabelSegment{{6, 5, 4}, {0.75, 0.75, 0.75}, false, -1, 0},
    };
    const std::vector<ri_meshing::Vec3f> vertices(15);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths,
            {old_segments},
            vertices,
            no_incidence,
            no_incidence,
            config,
            no_proxy,
            nullptr,
            &old_paths,
        },
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_closed_loop_with_single_seam_omission()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7, 8}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1, 2, 3}, {0.0, 0.25, 0.50, 0.75}, true}
    };
    const std::vector<ri_meshing::Vec3f> vertices(9);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_closed_loop_with_two_seam_omissions()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7, 8, 9}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1, 2, 3}, {0.0, 0.20, 0.40, 0.60}, true}
    };
    const std::vector<ri_meshing::Vec3f> vertices(10);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_closed_loop_with_three_cyclic_seam_omissions()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7, 8, 9, 10, 11, 12}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1, 2, 3}, {0.50, 0.70, 0.90, 1.125}, true}
    };
    const std::vector<ri_meshing::Vec3f> vertices(13);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_closed_loop_with_four_cyclic_seam_omissions()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1, 2, 3}, {0.0, 0.15, 0.35, 0.55}, true}
    };
    const std::vector<ri_meshing::Vec3f> vertices(14);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_overlapping_old_source_candidates()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1, 2, 3}, {0.0, 0.333333, 0.666667, 1.0}, false, -1, 0},
        OldLabelSegment{{8, 9, 10, 11}, {0.0, 0.333333, 0.666667, 1.0}, false, -1, 1},
    };
    const std::vector<ri_meshing::Vec3f> vertices(12);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths, {old_segments}, vertices, no_incidence, no_incidence, config, no_proxy},
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_adjacent_segment_bridge_candidates()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> old_paths{{{0, 1, 2, 3}, false}};
    const std::vector<BoundaryPath> new_paths{{{4, 5, 6, 7, 8, 9, 10}, true}};
    const std::vector<OldLabelSegment> old_segments{
        OldLabelSegment{{0, 1}, {0.333333, 0.40}, false, 0, 0},
        OldLabelSegment{{2, 3}, {0.60, 0.666667}, false, 1, 0},
    };
    const std::vector<ri_meshing::Vec3f> vertices(11);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths,
            {old_segments},
            vertices,
            no_incidence,
            no_incidence,
            config,
            no_proxy,
            nullptr,
            &old_paths,
        },
        nullptr,
        nullptr
    );
}

std::vector<ri_meshing::Triangle> generate_junction_closure_candidates()
{
    using ri_meshing::BoundaryPath;
    using ri_meshing::stitch_detail::CandidateGenerationRequest;
    using ri_meshing::stitch_detail::OldLabelSegment;

    const std::vector<BoundaryPath> old_paths{
        BoundaryPath{{0, 1}, false},
        BoundaryPath{{1, 2}, false},
    };
    const std::vector<BoundaryPath> new_paths{
        BoundaryPath{{3, 4, 5}, true},
        BoundaryPath{{6, 7, 8}, true},
    };
    const std::vector<OldLabelSegment> first_cluster{
        OldLabelSegment{{0, 1}, {0.0, 0.5}, false, -1, 0},
        OldLabelSegment{{1, 2}, {0.5, 1.0}, false, -1, 1},
    };
    const std::vector<OldLabelSegment> second_cluster{
        OldLabelSegment{{0, 1}, {0.0, 0.5}, false, -1, 0},
        OldLabelSegment{{1, 2}, {0.5, 1.0}, false, -1, 1},
    };
    const std::vector<ri_meshing::Vec3f> vertices(9);
    const ri_meshing::stitch_detail::BoundaryIncidenceData no_incidence;
    const ri_meshing::SparseMeshingConfig config;
    const std::function<void(const std::vector<std::array<int, 2>>&)> no_proxy;
    return ri_meshing::stitch_detail::generate_stitch_candidates(
        CandidateGenerationRequest{
            new_paths,
            {first_cluster, second_cluster},
            vertices,
            no_incidence,
            no_incidence,
            config,
            no_proxy,
            nullptr,
            &old_paths,
        },
        nullptr,
        nullptr
    );
}

void test_closed_loop_adds_seam_faces()
{
    const auto faces = generate_square_candidates(true);
    assert(has_face_with_vertices(faces, {0, 3, 4}));
    assert(has_face_with_vertices(faces, {3, 4, 7}));
}

void test_open_segment_does_not_add_seam_faces()
{
    const auto faces = generate_square_candidates(false);
    assert(!has_face_with_vertices(faces, {0, 3, 4}));
    assert(!has_face_with_vertices(faces, {3, 4, 7}));
}

void test_partial_old_segment_does_not_use_new_path_closure()
{
    const auto faces = generate_partial_arc_candidates();
    assert(!has_face_with_vertices(faces, {3, 6, 7}));
    assert(!has_face_with_vertices(faces, {3, 7, 4}));
}

void test_tied_partial_old_segment_stays_local_to_new_arc()
{
    const auto faces = generate_tied_partial_arc_candidates();
    assert(!has_face_with_vertices_pair(faces, 0, 3));
    assert(!has_face_with_vertices_pair(faces, 0, 10));
}

void test_tied_segment_uses_geometric_direction()
{
    const auto faces = generate_tied_reversed_segment_candidates();
    assert(has_face_with_vertices(faces, {0, 6, 7}));
    assert(has_face_with_vertices(faces, {0, 1, 7}));
    assert(has_face_with_vertices(faces, {1, 7, 8}));
    assert(has_face_with_vertices(faces, {1, 2, 8}));
    assert(!has_face_with_vertices(faces, {1, 2, 7}));
}

void test_collapsed_closed_seam_adds_local_closure_face()
{
    const auto faces = generate_collapsed_closed_seam_candidate();
    assert(has_face_with_vertices(faces, {0, 6, 12}));
    assert(!has_face_with_vertices_pair(faces, 6, 13));
}

void test_stitching_ignores_standalone_area_and_quality_thresholds()
{
    using ri_meshing::SparseMeshingConfig;
    using ri_meshing::Triangle;
    using ri_meshing::Vec3f;
    using ri_meshing::stitch_detail::ManifoldSelectionRequest;

    const std::vector<Vec3f> vertices{
        Vec3f{0.00F, 0.00F, 0.00F},
        Vec3f{0.03F, 0.00F, 0.00F},
        Vec3f{0.00F, 0.10F, 0.00F},
        Vec3f{0.03F, 0.10F, 0.00F},
    };
    const std::vector<Triangle> candidates{{0, 1, 2}};
    const std::vector<std::array<int, 2>> old_boundary_edges{{0, 1}};
    const std::vector<std::array<int, 2>> new_boundary_edges{{2, 3}};
    SparseMeshingConfig config;
    config.min_triangle_quality = 1.0;
    config.min_triangle_area = 1.0;

    const auto result = ri_meshing::stitch_detail::select_stitch_faces(
        ManifoldSelectionRequest{
            candidates,
            vertices,
            old_boundary_edges,
            new_boundary_edges,
            2,
            nullptr,
        },
        config,
        nullptr,
        nullptr
    );
    assert(has_face_with_vertices(result.faces, {0, 1, 2}));
}

void test_closed_loop_restores_single_new_seam_sample()
{
    const auto faces = generate_closed_loop_with_single_seam_omission();
    assert(has_face_with_vertices(faces, {3, 7, 8}));
    assert(has_face_with_vertices(faces, {3, 8, 4}));
    assert(has_face_with_vertices(faces, {3, 4, 0}));
}

void test_closed_loop_restores_two_new_seam_samples()
{
    const auto faces = generate_closed_loop_with_two_seam_omissions();
    assert(has_face_with_vertices(faces, {3, 7, 8}));
    assert(has_face_with_vertices(faces, {3, 8, 9}));
    assert(has_face_with_vertices(faces, {3, 9, 4}));
    assert(has_face_with_vertices(faces, {3, 4, 0}));
}

void test_closed_loop_restores_three_cyclic_seam_samples()
{
    const auto faces = generate_closed_loop_with_three_cyclic_seam_omissions();
    assert(has_face_with_vertices(faces, {3, 4, 5}));
    assert(has_face_with_vertices(faces, {3, 5, 6}));
    assert(has_face_with_vertices(faces, {3, 6, 7}));
}

void test_closed_loop_restores_four_cyclic_seam_samples()
{
    const auto faces = generate_closed_loop_with_four_cyclic_seam_omissions();
    assert(has_face_with_vertices(faces, {3, 9, 10}));
    assert(has_face_with_vertices(faces, {3, 10, 11}));
    assert(has_face_with_vertices(faces, {3, 11, 12}));
    assert(has_face_with_vertices(faces, {3, 12, 13}));
}

void test_overlapping_old_sources_do_not_share_new_boundary()
{
    const auto faces = generate_overlapping_old_source_candidates();
    bool uses_first_source = false;
    bool uses_second_source = false;
    for (const ri_meshing::Triangle face : faces) {
        for (const int vertex : {face.v0, face.v1, face.v2}) {
            uses_first_source = uses_first_source || (vertex >= 0 && vertex < 4);
            uses_second_source = uses_second_source || (vertex >= 8 && vertex < 12);
        }
    }
    assert(uses_first_source != uses_second_source);
}

void test_adjacent_segment_bridge_replaces_the_local_gap()
{
    const auto faces = generate_adjacent_segment_bridge_candidates();
    assert(has_face_with_vertices(faces, {1, 2, 7}));
}

void test_junction_closure_repairs_each_new_side()
{
    const auto faces = generate_junction_closure_candidates();
    assert(has_face_with_vertices(faces, {0, 1, 4}));
    assert(has_face_with_vertices(faces, {1, 4, 5}));
    assert(has_face_with_vertices(faces, {1, 2, 5}));
    assert(has_face_with_vertices(faces, {0, 1, 7}));
    assert(has_face_with_vertices(faces, {1, 7, 8}));
    assert(has_face_with_vertices(faces, {1, 2, 8}));
}

}  // namespace

int main()
{
    test_closed_loop_adds_seam_faces();
    test_open_segment_does_not_add_seam_faces();
    test_partial_old_segment_does_not_use_new_path_closure();
    test_tied_partial_old_segment_stays_local_to_new_arc();
    test_tied_segment_uses_geometric_direction();
    test_collapsed_closed_seam_adds_local_closure_face();
    test_stitching_ignores_standalone_area_and_quality_thresholds();
    test_closed_loop_restores_single_new_seam_sample();
    test_closed_loop_restores_two_new_seam_samples();
    test_closed_loop_restores_three_cyclic_seam_samples();
    test_closed_loop_restores_four_cyclic_seam_samples();
    test_overlapping_old_sources_do_not_share_new_boundary();
    test_adjacent_segment_bridge_replaces_the_local_gap();
    test_junction_closure_repairs_each_new_side();
    return 0;
}
