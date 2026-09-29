#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ri_meshing {

using BoundaryEdge = std::array<int, 2>;

struct BoundaryPath {
    std::vector<int> ids;
    bool closed{};
};

struct BoundaryNeighborRecord {
    int vertex{};
    std::uint8_t forward{};
};

struct BoundaryPathExtractionWorkspace {
    std::vector<int> dense_lookup;
    std::vector<int> degree_counts;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> cursors;
    std::vector<BoundaryNeighborRecord> neighbors;
    std::vector<std::uint8_t> visited;
    std::vector<std::pair<int, int>> direction_votes;
};

struct BoundaryPathExtractionStats {
    double edge_records_milliseconds{};
    double adjacency_milliseconds{};
    double orientation_milliseconds{};
    double final_sort_milliseconds{};
    double vertex_catalog_milliseconds{};
    double vertex_lookup_milliseconds{};
    double csr_index_milliseconds{};
    double csr_offsets_milliseconds{};
    double csr_fill_milliseconds{};
    double csr_sort_milliseconds{};
    double csr_visited_milliseconds{};
    double dirty_collection_milliseconds{};
    double cache_publish_milliseconds{};
    std::size_t input_edges{};
    std::size_t unique_edges{};
    std::size_t undirected_paths{};
    std::size_t oriented_paths{};
    std::size_t dirty_seed_vertices{};
    std::size_t local_boundary_vertices{};
    std::size_t local_boundary_edges{};
    std::size_t affected_path_tokens{};
    std::size_t replacement_paths{};
    std::size_t cached_paths_before{};
    std::size_t cached_paths_after{};
    std::size_t seeded_walk_vertices{};
    std::size_t seeded_walk_steps{};
};

std::vector<BoundaryPath> extract_boundary_paths(const std::vector<BoundaryEdge>& edges);

// Extract simple cycles and recover their orientation from surviving edges.
// Ambiguous cycles with no directional majority are omitted.
std::vector<BoundaryPath> extract_oriented_boundary_paths(
    const std::vector<BoundaryEdge>& edges
);

std::vector<BoundaryPath> extract_oriented_boundary_paths(
    const std::vector<BoundaryEdge>& edges,
    BoundaryPathExtractionStats* stats
);

// Fast path for BoundaryGraph's sorted, unique boundary edge invariant.
std::vector<BoundaryPath> extract_oriented_boundary_paths_sorted_unique(
    const std::vector<BoundaryEdge>& edges,
    BoundaryPathExtractionStats* stats
);

std::vector<BoundaryPath> extract_oriented_boundary_paths_sorted_unique(
    const std::vector<BoundaryEdge>& edges,
    const std::vector<int>& vertex_ids,
    BoundaryPathExtractionStats* stats
);

std::vector<BoundaryPath> extract_oriented_boundary_paths_sorted_unique(
    const std::vector<BoundaryEdge>& edges,
    const std::vector<int>& vertex_ids,
    BoundaryPathExtractionStats* stats,
    BoundaryPathExtractionWorkspace* workspace
);

void extract_oriented_boundary_paths_sorted_unique_into(
    const std::vector<BoundaryEdge>& edges,
    const std::vector<int>& vertex_ids,
    BoundaryPathExtractionStats* stats,
    BoundaryPathExtractionWorkspace* workspace,
    std::vector<BoundaryPath>& output
);

void extract_boundary_paths(
    const std::vector<BoundaryEdge>& edges,
    std::vector<BoundaryPath>& output
);

void extract_boundary_paths(
    const std::vector<BoundaryEdge>& edges,
    std::vector<BoundaryPath>& output,
    BoundaryPathExtractionWorkspace* workspace
);

}  // namespace ri_meshing
