#pragma once

#include "ri_meshing/boundary_stitching.hpp"
#include "ri_meshing/boundary_paths.hpp"
#include "ri_meshing/config.hpp"
#include "ri_meshing/types.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace ri_meshing {

struct PipelineTiming;

struct HoleFillRequest {
    const std::vector<Vec3f>& vertices_global;
    const std::vector<BoundaryPath>& boundary_paths;
    const StitchConnectionMask* seam_mask{};
    const SparseMeshingConfig& config;
    int old_vertex_count{};
    Mat4f global_to_local{};
    PipelineTiming* timing{};
    bool collect_loop_metrics{};
};

struct HoleFillLoopMetrics {
    std::size_t path_index{};
    std::size_t edge_count{};
    MeshScalar perimeter_3d_m{std::numeric_limits<MeshScalar>::quiet_NaN()};
    MeshScalar ri_area_pixels{std::numeric_limits<MeshScalar>::quiet_NaN()};
    MeshScalar diameter_3d_m{std::numeric_limits<MeshScalar>::quiet_NaN()};
    bool closed{};
    bool mixed_old_new{};
    bool basic_valid{};
    bool perimeter_valid{};
    bool projection_valid{};
    bool ri_area_valid{};
    bool seam_supported{};
    bool hole_winding{};
    bool diameter_exact{};
    bool center_attempted{};
    bool center_accepted{};
    bool fallback_attempted{};
    bool fallback_accepted{};
    bool accepted{};
    std::size_t generated_faces{};
};

struct HoleFillResult {
    std::vector<Vec3f> vertices_global;
    std::vector<Triangle> faces;
    std::size_t candidate_loops{};
    std::size_t accepted_loops{};
    std::size_t support_loops{};
    std::size_t center_attempts{};
    std::size_t center_accepted{};
    std::size_t fallback_attempts{};
    std::size_t fallback_accepted{};
    std::vector<HoleFillLoopMetrics> loop_metrics;
};

HoleFillResult fill_local_seam_holes(const HoleFillRequest& request);

}  // namespace ri_meshing
