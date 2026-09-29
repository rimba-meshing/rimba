#pragma once

#include "ri_meshing/config.hpp"
#include "ri_meshing/boundary_paths.hpp"
#include "ri_meshing/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ri_meshing {

struct PipelineTiming;

struct StitchDebugPath {
    std::vector<int> ids;
    bool closed{};
};

struct StitchDebugLabel {
    int vertex_id{};
    int cluster{-1};
    int label{-1};
    MeshScalar path_t{-1.0F};
    MeshScalar score{};
};

struct StitchDebugSegment {
    int cluster{-1};
    int segment{};
    std::vector<int> ids;
    std::vector<MeshScalar> labels;
};

struct StitchConnectionMask {
    int height{};
    int width{};
    std::vector<std::uint8_t> pixels{};
    std::vector<std::uint8_t> old_support{};
    // Accepted post-overlap new pixels; rejected candidates are not support.
    std::vector<std::uint8_t> new_support{};

    [[nodiscard]] bool empty() const noexcept
    {
        return height <= 0 || width <= 0 || pixels.empty();
    }
};

struct StitchCoreDebug {
    std::vector<std::array<int, 2>> old_boundary_edges;
    std::vector<std::array<int, 2>> new_boundary_edges;
    std::vector<StitchDebugPath> old_paths;
    std::vector<StitchDebugPath> new_paths;
    std::vector<StitchDebugLabel> old_labels;
    std::vector<StitchDebugLabel> new_labels;
    std::vector<StitchDebugSegment> old_segments;
    std::vector<Triangle> candidate_faces_before_filter;
    std::vector<Triangle> candidate_faces_after_filter;
    std::vector<Triangle> selected_faces_local;
};

// Dense ownership storage reused by candidate generation. Generation tags
// make entries from previous frames invisible without clearing the whole
// compact vertex range on every frame.
struct CandidateOwnershipWorkspace {
    std::vector<int> owners;
    std::vector<std::uint32_t> generations;
    std::uint32_t generation{};
};

// Non-owning form used by the CUDA transaction. The core only reads its
// boundary input, so keeping the vectors in the transaction workspace allows
// their capacity to be reused across frames.
struct StitchCoreInput {
    const std::vector<Vec3f>& vertices_local;
    const std::vector<Vec3f>& vertex_normals_local;
    const std::vector<std::array<int, 2>>& old_boundary_edges;
    const std::vector<std::array<int, 2>>& new_boundary_edges;
    PipelineTiming* timing{};
    int old_vertex_count{};
    StitchCoreDebug* debug{};
    std::function<void(const std::vector<std::array<int, 2>>&)> stitch_proxy_edges_callback{};
    const StitchConnectionMask* connection_mask{};
    BoundaryPathExtractionWorkspace* path_workspace{};
    CandidateOwnershipWorkspace* ownership_workspace{};
};

struct StitchCoreResult {
    std::vector<Triangle> faces;
    std::size_t old_boundary_edges_used{};
    std::size_t old_boundary_vertices{};
    std::size_t new_boundary_vertices{};
    std::size_t candidate_faces{};
};

StitchCoreResult stitch_core(
    const StitchCoreInput& input,
    const SparseMeshingConfig& config
);

}  // namespace ri_meshing
