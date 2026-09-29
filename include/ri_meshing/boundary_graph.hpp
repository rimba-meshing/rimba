#pragma once

#include "ri_meshing/boundary_paths.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ri_meshing {

class BoundaryGraph {
public:
    using Edge = std::array<int, 2>;
    using EdgeBlock = std::vector<Edge>;
    using EdgeBlocks = std::vector<EdgeBlock>;

    void clear();
    std::vector<int> toggle_boundary_edges(const std::vector<Edge>& edges);

    const std::vector<Edge>& boundary_edges() const;
    const EdgeBlocks& boundary_edge_blocks() const;
    const std::vector<int>& boundary_vertex_ids() const;
    const std::vector<BoundaryPath>& oriented_boundary_paths() const;
    const std::vector<BoundaryPath>& oriented_boundary_paths(
        BoundaryPathExtractionStats* stats
    ) const;
    const std::vector<BoundaryPath>& oriented_boundary_paths_for_holes(
        const std::vector<Edge>& seed_edges,
        std::size_t max_path_edges,
        BoundaryPathExtractionStats* stats
    ) const;

    std::size_t boundary_edge_count() const;
    std::array<std::size_t, 4> incidence_histogram() const;

private:
    static std::uint64_t edge_key(int a, int b);
    static bool edge_less(Edge a, Edge b);
    static std::vector<Edge> sorted_odd_toggles(const std::vector<Edge>& edges);
    static void add_neighbor(
        std::unordered_map<int, std::vector<BoundaryNeighborRecord>>& neighbors,
        int vertex,
        int neighbor,
        bool forward
    );
    static void remove_neighbor(
        std::unordered_map<int, std::vector<BoundaryNeighborRecord>>& neighbors,
        int vertex,
        int neighbor
    );
    void collect_local_boundary(
        const std::vector<int>& seed_vertices,
        std::size_t max_hops,
        std::vector<Edge>& edges,
        std::vector<int>& vertex_ids
    ) const;
    std::uint32_t begin_local_boundary_marking() const;
    void add_boundary_edge(Edge edge);
    void remove_boundary_edge(Edge edge);
    void update_vertex_degrees(Edge edge, int delta, std::vector<int>& touched_vertices);
    std::vector<int> apply_vertex_membership_changes(std::vector<int>& touched_vertices);
    bool contains_boundary_edge(Edge edge) const;
    void apply_edge_toggles(const std::vector<Edge>& toggles);
    void rebuild_oriented_path_index() const;
    void rebuild_oriented_paths_full(BoundaryPathExtractionStats* stats) const;
    void rebuild_oriented_paths_incremental(BoundaryPathExtractionStats* stats) const;

    static constexpr std::size_t edge_block_target_size_ = 4096;
    EdgeBlocks boundary_edge_blocks_;
    mutable std::vector<Edge> flattened_boundary_edges_;
    mutable bool flattened_boundary_edges_valid_{};
    std::size_t boundary_edge_count_{};
    std::vector<int> sorted_boundary_vertex_ids_;
    std::vector<int> vertex_degrees_by_id_;
    std::unordered_map<int, std::vector<BoundaryNeighborRecord>> boundary_neighbors_;
    mutable std::vector<BoundaryPath> oriented_boundary_paths_cache_;
    mutable std::vector<BoundaryPath> hole_boundary_paths_cache_;
    mutable BoundaryPathExtractionWorkspace oriented_path_workspace_;
    mutable std::unordered_map<int, std::vector<std::uint64_t>> oriented_paths_by_vertex_;
    mutable std::vector<std::uint64_t> oriented_path_tokens_;
    mutable std::unordered_map<std::uint64_t, std::size_t> oriented_path_indices_;
    mutable std::uint64_t next_oriented_path_token_{1};
    mutable std::vector<int> dirty_boundary_vertices_;
    mutable std::vector<std::uint32_t> local_boundary_marks_;
    mutable std::vector<int> local_boundary_queue_;
    mutable std::vector<std::size_t> local_boundary_queue_distances_;
    mutable std::vector<Edge> local_boundary_edges_;
    mutable std::vector<int> local_boundary_vertex_ids_;
    mutable std::uint32_t local_boundary_generation_{1};
    mutable bool oriented_path_cache_valid_{};
    mutable bool oriented_path_cache_snapshot_{};
};

}  // namespace ri_meshing
