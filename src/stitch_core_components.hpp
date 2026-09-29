#pragma once

#include "ri_meshing/boundary_paths.hpp"
#include "ri_meshing/boundary_stitching.hpp"
#include "stitch_connection.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ri_meshing::stitch_detail {

using Edge = std::array<int, 2>;

struct LabelHit {
    int cluster{-1};
    int label{-1};
    MeshScalar path_t{-1.0F};
    MeshScalar score{std::numeric_limits<MeshScalar>::infinity()};
    int component{-1};
};

struct OldLabelSegment {
    std::vector<int> ids;
    std::vector<MeshScalar> labels;
    bool closed{};
    int component{-1};
    int source_path{-1};
};

struct Vec2d {
    MeshScalar x{};
    MeshScalar y{};
};

struct BoundaryIncidence {
    int neighbor{-1};
    Vec3f tangent{};
    Vec3f open_dir{};
    Vec2d angular_tangent{};
    Vec2d angular_open_dir{};
    int frontier_component{-1};
};

struct BoundaryIncidenceData {
    std::unordered_map<int, std::vector<BoundaryIncidence>> by_vertex;
    std::unordered_map<std::uint64_t, Vec3f> segment_open_dirs;
    bool frontier_components_annotated{};
};

struct BoundaryMatchingRequest {
    const std::vector<Vec3f>& vertices;
    const std::vector<Edge>& old_boundary_edges;
    const std::vector<Edge>& new_boundary_edges;
    const std::vector<Vec3f>& vertex_normals;
    const std::vector<BoundaryPath>& old_paths;
    const std::vector<BoundaryPath>& new_paths;
    int old_vertex_count{};
    bool old_ids_are_dense{};
    StitchConnectionOracle* connection_oracle{};
};

struct BoundaryMatchingResult {
    std::vector<std::vector<OldLabelSegment>> old_segments_by_cluster;
    BoundaryIncidenceData old_incidence;
    BoundaryIncidenceData new_incidence;
};

struct CandidateGenerationRequest {
    const std::vector<BoundaryPath>& new_paths;
    const std::vector<std::vector<OldLabelSegment>>& old_segments_by_cluster;
    const std::vector<Vec3f>& vertices;
    const BoundaryIncidenceData& old_incidence;
    const BoundaryIncidenceData& new_incidence;
    const SparseMeshingConfig& config;
    const std::function<void(const std::vector<Edge>&)>& proxy_edges_callback;
    const StitchConnectionOracle* connection_oracle{};
    const std::vector<BoundaryPath>* old_paths{};
    CandidateOwnershipWorkspace* ownership_workspace{};
};

struct ManifoldSelectionRequest {
    std::vector<Triangle> candidates;
    const std::vector<Vec3f>& vertices;
    const std::vector<Edge>& old_boundary_edges;
    const std::vector<Edge>& new_boundary_edges;
    int old_vertex_count{};
    const StitchConnectionOracle* connection_oracle{};
};

struct ManifoldSelectionResult {
    std::vector<Triangle> faces;
    std::size_t candidate_faces{};
};

BoundaryMatchingResult match_boundaries(
    const BoundaryMatchingRequest& request,
    const SparseMeshingConfig& config,
    StitchCoreDebug* debug,
    PipelineTiming* timing
);

std::vector<Triangle> generate_stitch_candidates(
    const CandidateGenerationRequest& request,
    StitchCoreDebug* debug,
    PipelineTiming* timing
);

ManifoldSelectionResult select_stitch_faces(
    ManifoldSelectionRequest request,
    const SparseMeshingConfig& config,
    StitchCoreDebug* debug,
    PipelineTiming* timing
);

}  // namespace ri_meshing::stitch_detail
