#pragma once

#include "ri_meshing/cuda_meshing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ri_meshing {

struct CompactStitchInput {
    std::vector<Vec3f> vertices_local;
    std::vector<Vec3f> vertex_normals_local;
    std::vector<std::array<int, 2>> old_boundary_edges;
    std::vector<std::array<int, 2>> new_boundary_edges;
    std::vector<int> compact_to_global;
    int old_vertex_count{};
};

struct FilteredOldBoundaryEdges {
    std::vector<std::array<int, 2>> edges;
    std::vector<int> old_vertex_ids;
    std::vector<int> new_vertex_ids;
};

struct ProjectedRiPoint {
    int row{};
    int col{};
    float range{};
    bool valid{};
};

struct StitchProxyScratch {
    StitchProxyMetrics metrics;
    std::vector<std::uint8_t> proxy_mask;
    std::vector<float> proxy_depth;
};

struct StitchProxyEdgeSet {
    std::vector<std::array<int, 2>> old_edges;
    std::vector<std::array<int, 2>> new_edges;
    std::vector<std::array<int, 2>> bridge_edges;

    [[nodiscard]] bool empty() const noexcept
    {
        return old_edges.empty() && new_edges.empty() && bridge_edges.empty();
    }
};

struct StitchAppendResult {
    std::size_t stitch_faces{};
    std::size_t hole_faces{};
    StitchProxyMetrics proxy;
    StitchProxyEdgeSet proxy_edges;
    int proxy_base_face_count{};
};

struct StitchSpatialKey {
    int x{};
    int y{};
    int z{};

    bool operator==(StitchSpatialKey other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct StitchSpatialKeyHash {
    std::size_t operator()(StitchSpatialKey key) const noexcept
    {
        std::size_t seed = 0;
        const auto mix = [&](int value) {
            seed ^= std::hash<int>{}(value) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        };
        mix(key.x);
        mix(key.y);
        mix(key.z);
        return seed;
    }
};

struct StitchSpatialBoundaryIndex {
    MeshScalar cell_size{};
    bool initialized{};
    std::unordered_map<StitchSpatialKey, std::vector<int>, StitchSpatialKeyHash> active_cells;
    std::unordered_map<StitchSpatialKey, std::vector<int>, StitchSpatialKeyHash> new_cells;
    std::vector<StitchSpatialKey> vertex_cell_keys;
    std::vector<std::size_t> vertex_cell_positions;
    std::vector<std::uint8_t> active_boundary_flags;
    std::vector<std::uint8_t> near_old_vertex_flags;
    std::vector<std::uint8_t> filtered_old_vertex_flags;
    std::vector<int> active_boundary_ids;
    std::vector<int> near_old_vertex_ids;
    std::vector<int> filtered_old_vertex_ids;

    void clear()
    {
        cell_size = 0.0;
        initialized = false;
        active_cells.clear();
        new_cells.clear();
        vertex_cell_keys.clear();
        vertex_cell_positions.clear();
        active_boundary_flags.clear();
        near_old_vertex_flags.clear();
        filtered_old_vertex_flags.clear();
        active_boundary_ids.clear();
        near_old_vertex_ids.clear();
        filtered_old_vertex_ids.clear();
    }
};

}  // namespace ri_meshing
