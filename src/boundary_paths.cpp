#include "ri_meshing/boundary_paths.hpp"
#include "ri_meshing/timing.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ri_meshing {
namespace {

constexpr std::size_t missing_neighbor = std::numeric_limits<std::size_t>::max();

using NeighborRecord = BoundaryNeighborRecord;

bool reversed_is_smaller(const std::vector<int>& ids)
{
    return std::lexicographical_compare(ids.rbegin(), ids.rend(), ids.begin(), ids.end());
}

void canonicalize_closed_path(BoundaryPath& path)
{
    if (path.ids.empty()) {
        return;
    }
    const auto min_iter = std::min_element(path.ids.begin(), path.ids.end());
    std::rotate(path.ids.begin(), min_iter, path.ids.end());

    std::vector<int> reversed(path.ids.rbegin(), path.ids.rend());
    const auto reversed_min = std::min_element(reversed.begin(), reversed.end());
    std::rotate(reversed.begin(), reversed_min, reversed.end());
    if (std::lexicographical_compare(reversed.begin(), reversed.end(), path.ids.begin(), path.ids.end())) {
        path.ids = std::move(reversed);
    }
}

void canonicalize_path(BoundaryPath& path)
{
    if (path.ids.size() <= 1) {
        return;
    }
    if (path.closed) {
        canonicalize_closed_path(path);
        return;
    }
    if (reversed_is_smaller(path.ids)) {
        std::reverse(path.ids.begin(), path.ids.end());
    }
}

bool path_less(const BoundaryPath& a, const BoundaryPath& b)
{
    const auto key = [](const BoundaryPath& path) {
        constexpr int missing = std::numeric_limits<int>::max();
        return std::tuple{
            path.ids.empty() ? missing : path.ids.front(),
            path.ids.size() > 1 ? path.ids[1] : missing,
            path.ids.empty() ? missing : path.ids.back(),
            path.closed,
            path.ids.size(),
        };
    };
    const auto key_a = key(a);
    const auto key_b = key(b);
    if (key_a != key_b) {
        return key_a < key_b;
    }
    return std::lexicographical_compare(a.ids.begin(), a.ids.end(), b.ids.begin(), b.ids.end());
}

void finalize_paths(std::vector<BoundaryPath>& paths)
{
    for (BoundaryPath& path : paths) {
        canonicalize_path(path);
    }
    std::sort(paths.begin(), paths.end(), path_less);
}

class BoundaryVertexCatalog {
public:
    explicit BoundaryVertexCatalog(const std::vector<BoundaryEdge>& edges)
    {
        collect(edges);
        if (!ids_.empty()) {
            normalize();
        }
    }

    bool empty() const { return ids_.empty(); }
    bool uses_dense_range() const { return use_dense_range_; }
    int min_id() const { return min_id_; }
    int max_id() const { return max_id_; }
    const std::vector<int>& ids() const { return ids_; }

    std::vector<int> make_dense_lookup() const
    {
        std::vector<int> lookup(id_range_, -1);
        for (std::size_t idx = 0; idx < ids_.size(); ++idx) {
            lookup[static_cast<std::size_t>(ids_[idx] - min_id_)] = static_cast<int>(idx);
        }
        return lookup;
    }

    std::unordered_map<int, int> make_sparse_lookup() const
    {
        std::unordered_map<int, int> lookup;
        lookup.reserve(ids_.size());
        for (std::size_t idx = 0; idx < ids_.size(); ++idx) {
            lookup.emplace(ids_[idx], static_cast<int>(idx));
        }
        return lookup;
    }

private:
    void collect(const std::vector<BoundaryEdge>& edges)
    {
        ids_.reserve(edges.size() * 2);
        for (const BoundaryEdge edge : edges) {
            if (edge[0] < 0 || edge[1] < 0) {
                continue;
            }
            ids_.push_back(edge[0]);
            ids_.push_back(edge[1]);
            min_id_ = std::min({min_id_, edge[0], edge[1]});
            max_id_ = std::max({max_id_, edge[0], edge[1]});
        }
    }

    void normalize()
    {
        const std::size_t raw_id_refs = ids_.size();
        id_range_ = static_cast<std::size_t>(max_id_ - min_id_) + 1;
        use_dense_range_ = id_range_ <= raw_id_refs * 4 + 1024;
        if (use_dense_range_) {
            normalize_dense();
        } else {
            std::sort(ids_.begin(), ids_.end());
            ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());
        }
    }

    void normalize_dense()
    {
        std::vector<std::uint8_t> present(id_range_, std::uint8_t{0});
        for (const int vertex_id : ids_) {
            present[static_cast<std::size_t>(vertex_id - min_id_)] = 1;
        }
        ids_.clear();
        ids_.reserve(id_range_);
        for (std::size_t offset = 0; offset < present.size(); ++offset) {
            if (present[offset] != 0) {
                ids_.push_back(min_id_ + static_cast<int>(offset));
            }
        }
    }

    std::vector<int> ids_;
    int min_id_{std::numeric_limits<int>::max()};
    int max_id_{-1};
    std::size_t id_range_{};
    bool use_dense_range_{};
};

class DensePathAdjacency {
public:
    DensePathAdjacency(
        const std::vector<BoundaryEdge>& edges,
        const std::vector<int>& vertex_ids,
        int min_id,
        int max_id,
        bool edges_are_unique,
        BoundaryPathExtractionStats* stats,
        BoundaryPathExtractionWorkspace& workspace
    )
        : vertex_ids_(vertex_ids),
          id_to_index_(workspace.dense_lookup),
          min_id_(min_id),
          max_id_(max_id),
          workspace_(workspace)
    {
        build(edges, edges_are_unique, stats);
    }

    std::size_t size() const { return vertex_ids_.size(); }
    std::size_t neighbor_count(std::size_t vertex) const
    {
        return workspace_.offsets[vertex + 1] - workspace_.offsets[vertex];
    }
    int neighbor_at(std::size_t vertex, std::size_t position) const
    {
        return workspace_.neighbors[workspace_.offsets[vertex] + position].vertex;
    }

    bool is_visited(int vertex, int neighbor) const
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return false;
        }
        const std::size_t index = neighbor_index(static_cast<std::size_t>(vertex), neighbor);
        return index != missing_neighbor && workspace_.visited[index] != 0;
    }

    bool is_forward(int vertex, int neighbor) const
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return false;
        }
        const std::size_t index = neighbor_index(static_cast<std::size_t>(vertex), neighbor);
        return index != missing_neighbor && workspace_.neighbors[index].forward != 0;
    }

    void mark_visited(int a, int b)
    {
        mark_half_edge(a, b);
        mark_half_edge(b, a);
    }

private:
    int vertex_index(int vertex_id) const
    {
        if (vertex_id < min_id_ || vertex_id > max_id_) {
            return -1;
        }
        return id_to_index_[static_cast<std::size_t>(vertex_id - min_id_)];
    }

    void build(
        const std::vector<BoundaryEdge>& edges,
        bool edges_are_unique,
        BoundaryPathExtractionStats* stats
    )
    {
        const Stopwatch index_stopwatch;
        workspace_.degree_counts.assign(size(), 0);
        count_degrees(edges, workspace_.degree_counts);
        if (stats != nullptr) {
            stats->csr_index_milliseconds = index_stopwatch.elapsed_milliseconds();
        }

        const Stopwatch offsets_stopwatch;
        workspace_.offsets.assign(size() + 1, 0);
        for (std::size_t idx = 0; idx < workspace_.degree_counts.size(); ++idx) {
            workspace_.offsets[idx + 1] = workspace_.offsets[idx]
                + static_cast<std::size_t>(workspace_.degree_counts[idx]);
        }
        if (stats != nullptr) {
            stats->csr_offsets_milliseconds = offsets_stopwatch.elapsed_milliseconds();
        }

        const Stopwatch fill_stopwatch;
        populate_neighbors(edges);
        if (stats != nullptr) {
            stats->csr_fill_milliseconds = fill_stopwatch.elapsed_milliseconds();
        }
        if (edges_are_unique) {
            const Stopwatch sort_stopwatch;
            sort_neighbors();
            if (stats != nullptr) {
                stats->csr_sort_milliseconds = sort_stopwatch.elapsed_milliseconds();
            }
            const Stopwatch visited_stopwatch;
            workspace_.visited.assign(workspace_.neighbors.size(), std::uint8_t{0});
            if (stats != nullptr) {
                stats->csr_visited_milliseconds = visited_stopwatch.elapsed_milliseconds();
            }
            return;
        }
        const Stopwatch dedup_stopwatch;
        deduplicate_neighbors();
        if (stats != nullptr) {
            stats->csr_sort_milliseconds = dedup_stopwatch.elapsed_milliseconds();
        }
    }

    void sort_neighbors()
    {
        for (std::size_t idx = 0; idx < size(); ++idx) {
            auto begin = workspace_.neighbors.begin()
                + static_cast<std::ptrdiff_t>(workspace_.offsets[idx]);
            auto end = workspace_.neighbors.begin()
                + static_cast<std::ptrdiff_t>(workspace_.offsets[idx + 1]);
            std::sort(begin, end, [](NeighborRecord first, NeighborRecord second) {
                return first.vertex < second.vertex;
            });
        }
    }

    void count_degrees(const std::vector<BoundaryEdge>& edges, std::vector<int>& counts) const
    {
        for (const BoundaryEdge edge : edges) {
            const int a = vertex_index(edge[0]);
            const int b = vertex_index(edge[1]);
            if (a < 0 || b < 0 || a == b) {
                continue;
            }
            ++counts[static_cast<std::size_t>(a)];
            ++counts[static_cast<std::size_t>(b)];
        }
    }

    void populate_neighbors(const std::vector<BoundaryEdge>& edges)
    {
        workspace_.neighbors.resize(workspace_.offsets.back());
        workspace_.cursors = workspace_.offsets;
        for (const BoundaryEdge edge : edges) {
            const int a = vertex_index(edge[0]);
            const int b = vertex_index(edge[1]);
            if (a < 0 || b < 0 || a == b) {
                continue;
            }
            const std::size_t first_cursor = workspace_.cursors[static_cast<std::size_t>(a)]++;
            const std::size_t second_cursor = workspace_.cursors[static_cast<std::size_t>(b)]++;
            workspace_.neighbors[first_cursor] = NeighborRecord{
                b,
                static_cast<std::uint8_t>(
                    edge[0] == vertex_ids_[static_cast<std::size_t>(a)]
                    && edge[1] == vertex_ids_[static_cast<std::size_t>(b)]
                )
            };
            workspace_.neighbors[second_cursor] = NeighborRecord{a, 0};
        }
    }

    void deduplicate_neighbors()
    {
        std::vector<std::size_t> unique_offsets(size() + 1, 0);
        std::vector<NeighborRecord> unique_neighbors;
        unique_neighbors.reserve(workspace_.neighbors.size());
        for (std::size_t idx = 0; idx < size(); ++idx) {
            auto begin = workspace_.neighbors.begin()
                + static_cast<std::ptrdiff_t>(workspace_.offsets[idx]);
            auto end = workspace_.neighbors.begin()
                + static_cast<std::ptrdiff_t>(workspace_.offsets[idx + 1]);
            std::sort(begin, end, [](NeighborRecord first, NeighborRecord second) {
                return first.vertex < second.vertex;
            });
            end = std::unique(begin, end, [](NeighborRecord first, NeighborRecord second) {
                return first.vertex == second.vertex;
            });
            unique_offsets[idx] = unique_neighbors.size();
            unique_neighbors.insert(unique_neighbors.end(), begin, end);
        }
        unique_offsets.back() = unique_neighbors.size();
        workspace_.offsets = std::move(unique_offsets);
        workspace_.neighbors = std::move(unique_neighbors);
        workspace_.visited.assign(workspace_.neighbors.size(), std::uint8_t{0});
    }

    std::size_t neighbor_index(std::size_t vertex, int neighbor) const
    {
        const std::size_t begin = workspace_.offsets[vertex];
        const std::size_t count = workspace_.offsets[vertex + 1] - begin;
        if (count == 1) {
            return workspace_.neighbors[begin].vertex == neighbor ? begin : missing_neighbor;
        }
        if (count == 2) {
            if (workspace_.neighbors[begin].vertex == neighbor) {
                return begin;
            }
            return workspace_.neighbors[begin + 1].vertex == neighbor
                ? begin + 1 : missing_neighbor;
        }
        const auto first = workspace_.neighbors.begin()
            + static_cast<std::ptrdiff_t>(begin);
        const auto last = workspace_.neighbors.begin()
            + static_cast<std::ptrdiff_t>(workspace_.offsets[vertex + 1]);
        const auto iter = std::lower_bound(
            first, last, neighbor, [](NeighborRecord item, int value) {
                return item.vertex < value;
            }
        );
        return iter == last || iter->vertex != neighbor
            ? missing_neighbor
            : static_cast<std::size_t>(iter - workspace_.neighbors.begin());
    }

    void mark_half_edge(int vertex, int neighbor)
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return;
        }
        const std::size_t index = neighbor_index(static_cast<std::size_t>(vertex), neighbor);
        if (index != missing_neighbor) {
            workspace_.visited[index] = 1;
        }
    }

    const std::vector<int>& vertex_ids_;
    const std::vector<int>& id_to_index_;
    int min_id_;
    int max_id_;
    BoundaryPathExtractionWorkspace& workspace_;
};

struct PathAdjacency {
    std::vector<int> neighbors;
    std::vector<std::uint8_t> visited;
};

class SparsePathAdjacency {
public:
    SparsePathAdjacency(
        const std::vector<BoundaryEdge>& edges,
        const std::vector<int>& vertex_ids,
        const std::unordered_map<int, int>& id_to_index,
        bool edges_are_unique
    )
        : adjacency_(vertex_ids.size())
    {
        build(edges, id_to_index, edges_are_unique);
    }

    std::size_t size() const { return adjacency_.size(); }
    std::size_t neighbor_count(std::size_t vertex) const { return adjacency_[vertex].neighbors.size(); }
    int neighbor_at(std::size_t vertex, std::size_t position) const { return adjacency_[vertex].neighbors[position]; }

    bool is_visited(int vertex, int neighbor) const
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return false;
        }
        const PathAdjacency& item = adjacency_[static_cast<std::size_t>(vertex)];
        const std::size_t index = neighbor_index(item.neighbors, neighbor);
        return index != missing_neighbor && item.visited[index] != 0;
    }

    bool is_forward(int, int) const
    {
        return false;
    }

    void mark_visited(int a, int b)
    {
        mark_half_edge(a, b);
        mark_half_edge(b, a);
    }

private:
    void build(
        const std::vector<BoundaryEdge>& edges,
        const std::unordered_map<int, int>& lookup,
        bool edges_are_unique
    )
    {
        for (const BoundaryEdge edge : edges) {
            const int a = vertex_index(lookup, edge[0]);
            const int b = vertex_index(lookup, edge[1]);
            if (a < 0 || b < 0 || a == b) {
                continue;
            }
            adjacency_[static_cast<std::size_t>(a)].neighbors.push_back(b);
            adjacency_[static_cast<std::size_t>(b)].neighbors.push_back(a);
        }
        for (PathAdjacency& item : adjacency_) {
            std::sort(item.neighbors.begin(), item.neighbors.end());
            if (!edges_are_unique) {
                item.neighbors.erase(
                    std::unique(item.neighbors.begin(), item.neighbors.end()), item.neighbors.end()
                );
            }
            item.visited.assign(item.neighbors.size(), std::uint8_t{0});
        }
    }

    static int vertex_index(const std::unordered_map<int, int>& lookup, int vertex_id)
    {
        const auto found = lookup.find(vertex_id);
        return found == lookup.end() ? -1 : found->second;
    }

    static std::size_t neighbor_index(const std::vector<int>& neighbors, int vertex)
    {
        if (neighbors.size() == 1) {
            return neighbors[0] == vertex ? std::size_t{0} : missing_neighbor;
        }
        if (neighbors.size() == 2) {
            if (neighbors[0] == vertex) {
                return std::size_t{0};
            }
            return neighbors[1] == vertex ? std::size_t{1} : missing_neighbor;
        }
        const auto iter = std::lower_bound(neighbors.begin(), neighbors.end(), vertex);
        return iter == neighbors.end() || *iter != vertex
            ? missing_neighbor
            : static_cast<std::size_t>(iter - neighbors.begin());
    }

    void mark_half_edge(int vertex, int neighbor)
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return;
        }
        PathAdjacency& item = adjacency_[static_cast<std::size_t>(vertex)];
        const std::size_t index = neighbor_index(item.neighbors, neighbor);
        if (index != missing_neighbor) {
            item.visited[index] = 1;
        }
    }

    std::vector<PathAdjacency> adjacency_;
};

class UniqueSparsePathAdjacency {
public:
    UniqueSparsePathAdjacency(
        const std::vector<BoundaryEdge>& edges,
        std::size_t vertex_count,
        const std::unordered_map<int, int>& id_to_index,
        BoundaryPathExtractionStats* stats
    )
        : vertex_count_(vertex_count)
    {
        build(edges, id_to_index, stats);
    }

    std::size_t size() const { return vertex_count_; }

    std::size_t neighbor_count(std::size_t vertex) const
    {
        return offsets_[vertex + 1] - offsets_[vertex];
    }

    int neighbor_at(std::size_t vertex, std::size_t position) const
    {
        return neighbors_[offsets_[vertex] + position].vertex;
    }

    bool is_visited(int vertex, int neighbor) const
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return false;
        }
        const std::size_t index = neighbor_index(static_cast<std::size_t>(vertex), neighbor);
        return index != missing_neighbor && visited_[index] != 0;
    }

    bool is_forward(int vertex, int neighbor) const
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return false;
        }
        const std::size_t index = neighbor_index(static_cast<std::size_t>(vertex), neighbor);
        return index != missing_neighbor && neighbors_[index].forward != 0;
    }

    void mark_visited(int a, int b)
    {
        mark_half_edge(a, b);
        mark_half_edge(b, a);
    }

private:
    void build(
        const std::vector<BoundaryEdge>& edges,
        const std::unordered_map<int, int>& lookup,
        BoundaryPathExtractionStats* stats
    )
    {
        const Stopwatch index_stopwatch;
        std::vector<std::size_t> degree_counts(vertex_count_, 0);
        std::vector<BoundaryEdge> indexed_edges;
        indexed_edges.reserve(edges.size());
        for (const BoundaryEdge edge : edges) {
            const int a = vertex_index(lookup, edge[0]);
            const int b = vertex_index(lookup, edge[1]);
            if (a < 0 || b < 0 || a == b) {
                continue;
            }
            ++degree_counts[static_cast<std::size_t>(a)];
            ++degree_counts[static_cast<std::size_t>(b)];
            indexed_edges.push_back(BoundaryEdge{a, b});
        }
        if (stats != nullptr) {
            stats->csr_index_milliseconds = index_stopwatch.elapsed_milliseconds();
        }

        const Stopwatch offsets_stopwatch;
        offsets_.assign(vertex_count_ + 1, 0);
        for (std::size_t index = 0; index < vertex_count_; ++index) {
            offsets_[index + 1] = offsets_[index] + degree_counts[index];
        }
        if (stats != nullptr) {
            stats->csr_offsets_milliseconds = offsets_stopwatch.elapsed_milliseconds();
        }

        const Stopwatch fill_stopwatch;
        neighbors_.resize(offsets_.back());
        std::vector<std::size_t> cursors = offsets_;
        for (const BoundaryEdge edge : indexed_edges) {
            const std::size_t first_cursor = cursors[static_cast<std::size_t>(edge[0])]++;
            const std::size_t second_cursor = cursors[static_cast<std::size_t>(edge[1])]++;
            neighbors_[first_cursor] = NeighborRecord{edge[1], 1};
            neighbors_[second_cursor] = NeighborRecord{edge[0], 0};
        }
        if (stats != nullptr) {
            stats->csr_fill_milliseconds = fill_stopwatch.elapsed_milliseconds();
        }

        const Stopwatch sort_stopwatch;
        for (std::size_t index = 0; index < vertex_count_; ++index) {
            auto first = neighbors_.begin() + static_cast<std::ptrdiff_t>(offsets_[index]);
            auto last = neighbors_.begin() + static_cast<std::ptrdiff_t>(offsets_[index + 1]);
            std::sort(first, last, [](NeighborRecord first_record, NeighborRecord second_record) {
                return first_record.vertex < second_record.vertex;
            });
        }
        if (stats != nullptr) {
            stats->csr_sort_milliseconds = sort_stopwatch.elapsed_milliseconds();
        }

        const Stopwatch visited_stopwatch;
        visited_.assign(neighbors_.size(), std::uint8_t{0});
        if (stats != nullptr) {
            stats->csr_visited_milliseconds = visited_stopwatch.elapsed_milliseconds();
        }
    }

    static int vertex_index(const std::unordered_map<int, int>& lookup, int vertex_id)
    {
        const auto found = lookup.find(vertex_id);
        return found == lookup.end() ? -1 : found->second;
    }

    std::size_t neighbor_index(std::size_t vertex, int neighbor) const
    {
        const std::size_t begin = offsets_[vertex];
        const std::size_t count = offsets_[vertex + 1] - begin;
        if (count == 1) {
            return neighbors_[begin].vertex == neighbor ? begin : missing_neighbor;
        }
        if (count == 2) {
            if (neighbors_[begin].vertex == neighbor) {
                return begin;
            }
            return neighbors_[begin + 1].vertex == neighbor ? begin + 1 : missing_neighbor;
        }
        const auto first = neighbors_.begin() + static_cast<std::ptrdiff_t>(begin);
        const auto last = neighbors_.begin() + static_cast<std::ptrdiff_t>(offsets_[vertex + 1]);
        const auto iter = std::lower_bound(
            first, last, neighbor, [](NeighborRecord item, int value) {
                return item.vertex < value;
            }
        );
        return iter == last || iter->vertex != neighbor
            ? missing_neighbor
            : static_cast<std::size_t>(iter - neighbors_.begin());
    }

    void mark_half_edge(int vertex, int neighbor)
    {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= size()) {
            return;
        }
        const std::size_t index = neighbor_index(static_cast<std::size_t>(vertex), neighbor);
        if (index != missing_neighbor) {
            visited_[index] = 1;
        }
    }

    std::size_t vertex_count_{};
    std::vector<std::size_t> offsets_;
    std::vector<NeighborRecord> neighbors_;
    std::vector<std::uint8_t> visited_;
};

template <typename Adjacency>
class BoundaryPathExtractor {
public:
    BoundaryPathExtractor(
        const std::vector<int>& vertex_ids,
        Adjacency& adjacency,
        std::size_t edge_count,
        std::vector<BoundaryPath>& paths,
        std::vector<std::pair<int, int>>* direction_votes = nullptr
    )
        : vertex_ids_(vertex_ids), adjacency_(adjacency), paths_(paths), direction_votes_(direction_votes)
    {
        paths_.clear();
        paths_.reserve(edge_count / 8);
        if (direction_votes_ != nullptr) {
            direction_votes_->clear();
            direction_votes_->reserve(edge_count / 8);
        }
    }

    void extract()
    {
        extract_branch_paths();
        extract_cycles();
        // The oriented caller consumes votes in extraction order and
        // canonicalizes the paths after applying those votes.
        if (direction_votes_ == nullptr) {
            finalize_paths(paths_);
        }
    }

private:
    void extract_branch_paths()
    {
        for (std::size_t vertex_idx = 0; vertex_idx < adjacency_.size(); ++vertex_idx) {
            if (adjacency_.neighbor_count(vertex_idx) == 2) {
                continue;
            }
            const int vertex = static_cast<int>(vertex_idx);
            for (std::size_t pos = 0; pos < adjacency_.neighbor_count(vertex_idx); ++pos) {
                const int next = adjacency_.neighbor_at(vertex_idx, pos);
                if (!adjacency_.is_visited(vertex, next)) {
                    consume_branch(vertex, next);
                }
            }
        }
    }

    void consume_branch(int start, int next)
    {
        BoundaryPath path;
        path.ids.push_back(vertex_ids_[static_cast<std::size_t>(start)]);
        int previous = start;
        int current = next;
        adjacency_.mark_visited(previous, current);
        std::pair<int, int> votes{};
        record_direction(previous, current, votes);
        while (true) {
            path.ids.push_back(vertex_ids_[static_cast<std::size_t>(current)]);
            if (adjacency_.neighbor_count(static_cast<std::size_t>(current)) != 2) {
                break;
            }
            const int first = adjacency_.neighbor_at(static_cast<std::size_t>(current), 0);
            const int candidate = first == previous ? adjacency_.neighbor_at(static_cast<std::size_t>(current), 1) : first;
            if (adjacency_.is_visited(current, candidate)) {
                break;
            }
            adjacency_.mark_visited(current, candidate);
            record_direction(current, candidate, votes);
            previous = current;
            current = candidate;
        }
        if (path.ids.size() >= 2) {
            paths_.push_back(std::move(path));
            record_path_votes(votes);
        }
    }

    void extract_cycles()
    {
        for (std::size_t vertex_idx = 0; vertex_idx < adjacency_.size(); ++vertex_idx) {
            const int vertex = static_cast<int>(vertex_idx);
            for (std::size_t pos = 0; pos < adjacency_.neighbor_count(vertex_idx); ++pos) {
                const int first_next = adjacency_.neighbor_at(vertex_idx, pos);
                if (!adjacency_.is_visited(vertex, first_next)) {
                    consume_cycle(vertex, first_next);
                }
            }
        }
    }

    void consume_cycle(int start, int first_next)
    {
        BoundaryPath path;
        int previous = start;
        int current = first_next;
        path.ids.push_back(vertex_ids_[static_cast<std::size_t>(start)]);
        adjacency_.mark_visited(previous, current);
        std::pair<int, int> votes{};
        record_direction(previous, current, votes);
        while (true) {
            path.ids.push_back(vertex_ids_[static_cast<std::size_t>(current)]);
            const std::size_t count = adjacency_.neighbor_count(static_cast<std::size_t>(current));
            if (count == 0) {
                break;
            }
            const int first = adjacency_.neighbor_at(static_cast<std::size_t>(current), 0);
            const int next = count == 1 || first != previous ? first : adjacency_.neighbor_at(static_cast<std::size_t>(current), 1);
            if (next == start) {
                record_direction(current, next, votes);
                path.closed = true;
                adjacency_.mark_visited(current, next);
                break;
            }
            if (adjacency_.is_visited(current, next)) {
                break;
            }
            adjacency_.mark_visited(current, next);
            record_direction(current, next, votes);
            previous = current;
            current = next;
        }
        if (path.ids.size() >= 2) {
            paths_.push_back(std::move(path));
            record_path_votes(votes);
        }
    }

    void record_direction(int from, int to, std::pair<int, int>& votes) const
    {
        if (direction_votes_ == nullptr) {
            return;
        }
        if (adjacency_.is_forward(from, to)) {
            ++votes.first;
        } else {
            ++votes.second;
        }
    }

    void record_path_votes(std::pair<int, int> votes)
    {
        if (direction_votes_ != nullptr) {
            direction_votes_->push_back(votes);
        }
    }

    const std::vector<int>& vertex_ids_;
    Adjacency& adjacency_;
    std::vector<BoundaryPath>& paths_;
    std::vector<std::pair<int, int>>* direction_votes_;
};

void extract_dense_paths(
    const std::vector<BoundaryEdge>& edges,
    const BoundaryVertexCatalog& vertices,
    std::vector<BoundaryPath>& output,
    bool edges_are_unique,
    BoundaryPathExtractionStats* stats,
    BoundaryPathExtractionWorkspace* workspace
)
{
    BoundaryPathExtractionWorkspace local_workspace;
    BoundaryPathExtractionWorkspace& scratch = workspace != nullptr ? *workspace : local_workspace;
    scratch.dense_lookup = vertices.make_dense_lookup();
    DensePathAdjacency adjacency(
        edges, vertices.ids(), vertices.min_id(), vertices.max_id(), edges_are_unique, stats, scratch
    );
    BoundaryPathExtractor extractor(vertices.ids(), adjacency, edges.size(), output);
    extractor.extract();
}

void extract_sparse_paths(
    const std::vector<BoundaryEdge>& edges,
    const BoundaryVertexCatalog& vertices,
    std::vector<BoundaryPath>& output,
    bool edges_are_unique,
    BoundaryPathExtractionStats* stats
)
{
    const Stopwatch lookup_stopwatch;
    const std::unordered_map<int, int> lookup = vertices.make_sparse_lookup();
    if (stats != nullptr) {
        stats->vertex_lookup_milliseconds = lookup_stopwatch.elapsed_milliseconds();
    }
    if (edges_are_unique) {
        UniqueSparsePathAdjacency adjacency(edges, vertices.ids().size(), lookup, stats);
        BoundaryPathExtractor extractor(vertices.ids(), adjacency, edges.size(), output);
        extractor.extract();
        return;
    }
    SparsePathAdjacency adjacency(edges, vertices.ids(), lookup, false);
    BoundaryPathExtractor extractor(vertices.ids(), adjacency, edges.size(), output);
    extractor.extract();
}

void extract_paths_internal(
    const std::vector<BoundaryEdge>& edges,
    std::vector<BoundaryPath>& output,
    bool edges_are_unique,
    BoundaryPathExtractionStats* stats = nullptr,
    BoundaryPathExtractionWorkspace* workspace = nullptr
)
{
    const Stopwatch catalog_stopwatch;
    const BoundaryVertexCatalog vertices(edges);
    if (stats != nullptr) {
        stats->vertex_catalog_milliseconds = catalog_stopwatch.elapsed_milliseconds();
    }
    if (vertices.empty()) {
        output.clear();
        return;
    }
    if (vertices.uses_dense_range()) {
        extract_dense_paths(edges, vertices, output, edges_are_unique, stats, workspace);
    } else {
        extract_sparse_paths(edges, vertices, output, edges_are_unique, stats);
    }
}

void extract_paths_internal_sorted_unique(
    const std::vector<BoundaryEdge>& edges,
    const std::vector<int>& vertex_ids,
    std::vector<BoundaryPath>& output,
    BoundaryPathExtractionStats* stats,
    std::vector<std::pair<int, int>>& direction_votes,
    BoundaryPathExtractionWorkspace* workspace
)
{
    if (vertex_ids.empty()) {
        output.clear();
        return;
    }
    const int min_id = vertex_ids.front();
    const int max_id = vertex_ids.back();
    const std::size_t id_range = static_cast<std::size_t>(max_id - min_id) + 1;
    constexpr std::size_t max_dense_lookup_bytes = 32U * 1024U * 1024U;
    const bool use_dense_range = id_range <= max_dense_lookup_bytes / sizeof(int);
    if (use_dense_range) {
        BoundaryPathExtractionWorkspace local_workspace;
        BoundaryPathExtractionWorkspace& scratch = workspace != nullptr ? *workspace : local_workspace;
        scratch.dense_lookup.assign(id_range, -1);
        for (std::size_t index = 0; index < vertex_ids.size(); ++index) {
            scratch.dense_lookup[static_cast<std::size_t>(vertex_ids[index] - min_id)] = static_cast<int>(index);
        }
        DensePathAdjacency adjacency(
            edges, vertex_ids, min_id, max_id, true, stats, scratch
        );
        BoundaryPathExtractor extractor(
            vertex_ids, adjacency, edges.size(), output, &direction_votes
        );
        extractor.extract();
        return;
    }

    const Stopwatch lookup_stopwatch;
    const std::unordered_map<int, int> lookup = [&vertex_ids]() {
        std::unordered_map<int, int> result;
        result.reserve(vertex_ids.size());
        for (std::size_t index = 0; index < vertex_ids.size(); ++index) {
            result.emplace(vertex_ids[index], static_cast<int>(index));
        }
        return result;
    }();
    if (stats != nullptr) {
        stats->vertex_lookup_milliseconds = lookup_stopwatch.elapsed_milliseconds();
    }
    UniqueSparsePathAdjacency adjacency(edges, vertex_ids.size(), lookup, stats);
    BoundaryPathExtractor extractor(
        vertex_ids, adjacency, edges.size(), output, &direction_votes
    );
    extractor.extract();
}

}  // namespace

std::vector<BoundaryPath> extract_boundary_paths(const std::vector<BoundaryEdge>& edges)
{
    std::vector<BoundaryPath> output;
    extract_boundary_paths(edges, output);
    return output;
}

namespace {

std::uint64_t undirected_edge_key(BoundaryEdge edge)
{
    const auto first = static_cast<std::uint32_t>(edge[0]);
    const auto second = static_cast<std::uint32_t>(edge[1]);
    const auto low = std::min(first, second);
    const auto high = std::max(first, second);
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

void rotate_closed_path_preserving_direction(BoundaryPath& path)
{
    if (path.ids.empty()) {
        return;
    }
    const auto minimum = std::min_element(path.ids.begin(), path.ids.end());
    std::rotate(path.ids.begin(), minimum, path.ids.end());
}

void orient_unique_paths_into(
    const std::vector<BoundaryEdge>& unique_edges,
    const std::vector<BoundaryEdge>& direction_edges,
    BoundaryPathExtractionStats* stats,
    const std::vector<int>* known_vertex_ids,
    BoundaryPathExtractionWorkspace* workspace,
    std::vector<BoundaryPath>& undirected_paths
)
{
    const bool instrument = stats != nullptr;
    const Stopwatch adjacency_stopwatch;
    std::vector<std::pair<int, int>> local_direction_votes;
    std::vector<std::pair<int, int>>& direction_votes = workspace != nullptr
        ? workspace->direction_votes : local_direction_votes;
    if (known_vertex_ids != nullptr) {
        extract_paths_internal_sorted_unique(
            unique_edges, *known_vertex_ids, undirected_paths, stats, direction_votes, workspace
        );
    } else {
        extract_paths_internal(unique_edges, undirected_paths, true, stats);
    }
    if (instrument) {
        stats->adjacency_milliseconds = adjacency_stopwatch.elapsed_milliseconds();
        stats->undirected_paths = undirected_paths.size();
    }

    const Stopwatch orientation_stopwatch;
    std::size_t oriented_path_count = 0;
    for (std::size_t path_index = 0; path_index < undirected_paths.size(); ++path_index) {
        BoundaryPath path = std::move(undirected_paths[path_index]);
        if (!path.closed || path.ids.size() < 3) {
            continue;
        }

        int forward_votes = 0;
        int reverse_votes = 0;
        if (known_vertex_ids != nullptr) {
            if (path_index < direction_votes.size()) {
                forward_votes = direction_votes[path_index].first;
                reverse_votes = direction_votes[path_index].second;
            }
        } else {
            for (std::size_t index = 0; index < path.ids.size(); ++index) {
                const BoundaryEdge path_edge{
                    path.ids[index], path.ids[(index + 1) % path.ids.size()]
                };
                const auto edge = std::lower_bound(
                    direction_edges.begin(), direction_edges.end(), undirected_edge_key(path_edge),
                    [](BoundaryEdge candidate, std::uint64_t key) {
                        return undirected_edge_key(candidate) < key;
                    }
                );
                if (edge == direction_edges.end()
                    || undirected_edge_key(*edge) != undirected_edge_key(path_edge)) {
                    continue;
                }
                if (*edge == path_edge) {
                    ++forward_votes;
                } else {
                    ++reverse_votes;
                }
            }
        }
        if (forward_votes == reverse_votes) {
            continue;
        }
        if (reverse_votes > forward_votes) {
            std::reverse(path.ids.begin(), path.ids.end());
        }
        rotate_closed_path_preserving_direction(path);
        undirected_paths[oriented_path_count++] = std::move(path);
    }
    undirected_paths.resize(oriented_path_count);
    if (instrument) {
        stats->orientation_milliseconds = orientation_stopwatch.elapsed_milliseconds();
        stats->oriented_paths = oriented_path_count;
    }

    const Stopwatch final_sort_stopwatch;
    std::sort(undirected_paths.begin(), undirected_paths.end(), path_less);
    if (instrument) {
        stats->final_sort_milliseconds = final_sort_stopwatch.elapsed_milliseconds();
    }
}

std::vector<BoundaryPath> orient_unique_paths(
    const std::vector<BoundaryEdge>& unique_edges,
    const std::vector<BoundaryEdge>& direction_edges,
    BoundaryPathExtractionStats* stats,
    const std::vector<int>* known_vertex_ids = nullptr,
    BoundaryPathExtractionWorkspace* workspace = nullptr
)
{
    std::vector<BoundaryPath> paths;
    orient_unique_paths_into(
        unique_edges, direction_edges, stats, known_vertex_ids, workspace, paths
    );
    return paths;
}

}  // namespace

std::vector<BoundaryPath> extract_oriented_boundary_paths(
    const std::vector<BoundaryEdge>& edges
)
{
    return extract_oriented_boundary_paths(edges, nullptr);
}

std::vector<BoundaryPath> extract_oriented_boundary_paths(
    const std::vector<BoundaryEdge>& edges,
    BoundaryPathExtractionStats* stats
)
{
    const bool instrument = stats != nullptr;
    if (instrument) {
        *stats = BoundaryPathExtractionStats{};
        stats->input_edges = edges.size();
    }
    const Stopwatch edge_records_stopwatch;
    struct EdgeRecord {
        BoundaryEdge first_edge{};
        std::size_t count{};
    };

    // Keep multiplicity and the first directed occurrence together. The
    // previous implementation maintained two hash tables with the same key.
    std::unordered_map<std::uint64_t, EdgeRecord> edge_records;
    edge_records.reserve(edges.size());
    for (const BoundaryEdge edge : edges) {
        if (edge[0] < 0 || edge[1] < 0 || edge[0] == edge[1]) {
            continue;
        }
        const std::uint64_t key = undirected_edge_key(edge);
        const auto [record, inserted] = edge_records.emplace(
            key, EdgeRecord{edge, 1}
        );
        if (!inserted) {
            ++record->second.count;
        }
    }

    std::vector<BoundaryEdge> unique_edges;
    unique_edges.reserve(edge_records.size());
    for (const auto& [key, record] : edge_records) {
        if (record.count == 1) {
            unique_edges.push_back(record.first_edge);
        }
    }
    std::sort(unique_edges.begin(), unique_edges.end(), [](BoundaryEdge a, BoundaryEdge b) {
        return undirected_edge_key(a) < undirected_edge_key(b);
    });
    if (instrument) {
        stats->edge_records_milliseconds = edge_records_stopwatch.elapsed_milliseconds();
        stats->unique_edges = unique_edges.size();
    }

    return orient_unique_paths(unique_edges, unique_edges, stats);
}

std::vector<BoundaryPath> extract_oriented_boundary_paths_sorted_unique(
    const std::vector<BoundaryEdge>& edges,
    BoundaryPathExtractionStats* stats
)
{
    if (stats != nullptr) {
        *stats = BoundaryPathExtractionStats{};
        stats->input_edges = edges.size();
        stats->unique_edges = edges.size();
    }
    // BoundaryGraph keeps this vector sorted by undirected edge key and
    // toggles edges by parity, so no multiplicity table or second sort is
    // needed here. The generic public extractor above remains available for
    // arbitrary, potentially duplicated input.
    return orient_unique_paths(edges, edges, stats);
}

std::vector<BoundaryPath> extract_oriented_boundary_paths_sorted_unique(
    const std::vector<BoundaryEdge>& edges,
    const std::vector<int>& vertex_ids,
    BoundaryPathExtractionStats* stats
)
{
    if (stats != nullptr) {
        *stats = BoundaryPathExtractionStats{};
        stats->input_edges = edges.size();
        stats->unique_edges = edges.size();
    }
    return extract_oriented_boundary_paths_sorted_unique(edges, vertex_ids, stats, nullptr);
}

std::vector<BoundaryPath> extract_oriented_boundary_paths_sorted_unique(
    const std::vector<BoundaryEdge>& edges,
    const std::vector<int>& vertex_ids,
    BoundaryPathExtractionStats* stats,
    BoundaryPathExtractionWorkspace* workspace
)
{
    if (stats != nullptr) {
        *stats = BoundaryPathExtractionStats{};
        stats->input_edges = edges.size();
        stats->unique_edges = edges.size();
    }
    return orient_unique_paths(edges, edges, stats, &vertex_ids, workspace);
}

void extract_oriented_boundary_paths_sorted_unique_into(
    const std::vector<BoundaryEdge>& edges,
    const std::vector<int>& vertex_ids,
    BoundaryPathExtractionStats* stats,
    BoundaryPathExtractionWorkspace* workspace,
    std::vector<BoundaryPath>& output
)
{
    if (stats != nullptr) {
        *stats = BoundaryPathExtractionStats{};
        stats->input_edges = edges.size();
        stats->unique_edges = edges.size();
    }
    orient_unique_paths_into(edges, edges, stats, &vertex_ids, workspace, output);
}

void extract_boundary_paths(
    const std::vector<BoundaryEdge>& edges,
    std::vector<BoundaryPath>& output
)
{
    extract_paths_internal(edges, output, false);
}

void extract_boundary_paths(
    const std::vector<BoundaryEdge>& edges,
    std::vector<BoundaryPath>& output,
    BoundaryPathExtractionWorkspace* workspace
)
{
    extract_paths_internal(edges, output, false, nullptr, workspace);
}

}  // namespace ri_meshing
