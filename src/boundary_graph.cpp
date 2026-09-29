#include "ri_meshing/boundary_graph.hpp"
#include "ri_meshing/timing.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <utility>

namespace {

bool boundary_path_less(
    const ri_meshing::BoundaryPath& first,
    const ri_meshing::BoundaryPath& second
)
{
    const auto key = [](const ri_meshing::BoundaryPath& path) {
        constexpr int missing = std::numeric_limits<int>::max();
        return std::tuple{
            path.ids.empty() ? missing : path.ids.front(),
            path.ids.size() > 1 ? path.ids[1] : missing,
            path.ids.empty() ? missing : path.ids.back(),
            path.closed,
            path.ids.size(),
        };
    };
    const auto first_key = key(first);
    const auto second_key = key(second);
    if (first_key != second_key) {
        return first_key < second_key;
    }
    return std::lexicographical_compare(
        first.ids.begin(), first.ids.end(), second.ids.begin(), second.ids.end()
    );
}

std::uint64_t hole_seed_edge_key(std::array<int, 2> edge)
{
    const auto low = static_cast<std::uint32_t>(std::min(edge[0], edge[1]));
    const auto high = static_cast<std::uint32_t>(std::max(edge[0], edge[1]));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

std::uint64_t sorted_edge_key(std::array<int, 2> edge)
{
    const auto low = static_cast<std::uint32_t>(std::min(edge[0], edge[1]));
    const auto high = static_cast<std::uint32_t>(std::max(edge[0], edge[1]));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

int compare_sorted_edges(
    ri_meshing::BoundaryGraph::Edge first,
    ri_meshing::BoundaryGraph::Edge second
)
{
    const std::uint64_t first_key = sorted_edge_key(first);
    const std::uint64_t second_key = sorted_edge_key(second);
    if (first_key < second_key) {
        return -1;
    }
    if (first_key > second_key) {
        return 1;
    }
    return 0;
}

int compare_sorted_ids(int first, int second)
{
    return first < second ? -1 : first > second ? 1 : 0;
}

template <typename Item, typename Compare>
void merge_sorted_toggles_in_place(
    std::vector<Item>& values,
    const std::vector<Item>& toggles,
    std::size_t additions,
    std::size_t removals,
    Compare compare
)
{
    const std::size_t old_size = values.size();
    const std::size_t new_size = old_size + additions - removals;
    if (additions != 0 && removals != 0) {
        std::vector<Item> merged;
        merged.reserve(new_size);
        std::size_t value_index = 0;
        std::size_t toggle_index = 0;
        while (value_index < old_size || toggle_index < toggles.size()) {
            if (value_index == old_size) {
                merged.push_back(toggles[toggle_index++]);
                continue;
            }
            if (toggle_index == toggles.size()) {
                merged.push_back(values[value_index++]);
                continue;
            }
            const Item& value = values[value_index];
            const Item& toggle = toggles[toggle_index];
            const int order = compare(value, toggle);
            if (order == 0) {
                ++value_index;
                ++toggle_index;
            } else if (order < 0) {
                merged.push_back(values[value_index++]);
            } else {
                merged.push_back(toggles[toggle_index++]);
            }
        }
        values.swap(merged);
        return;
    }
    if (new_size > old_size) {
        values.resize(new_size);
        std::size_t value_index = old_size;
        std::size_t toggle_index = toggles.size();
        std::size_t output_index = new_size;
        while (value_index > 0 || toggle_index > 0) {
            if (value_index == 0) {
                values[--output_index] = toggles[--toggle_index];
                continue;
            }
            if (toggle_index == 0) {
                values[--output_index] = values[--value_index];
                continue;
            }
            const Item& value = values[value_index - 1];
            const Item& toggle = toggles[toggle_index - 1];
            const int order = compare(value, toggle);
            if (order == 0) {
                --value_index;
                --toggle_index;
            } else if (order > 0) {
                values[--output_index] = values[--value_index];
            } else {
                values[--output_index] = toggles[--toggle_index];
            }
        }
        return;
    }

    std::size_t value_index = 0;
    std::size_t toggle_index = 0;
    std::size_t output_index = 0;
    while (value_index < old_size || toggle_index < toggles.size()) {
        if (value_index == old_size) {
            values[output_index++] = toggles[toggle_index++];
            continue;
        }
        if (toggle_index == toggles.size()) {
            values[output_index++] = values[value_index++];
            continue;
        }
        const Item& value = values[value_index];
        const Item& toggle = toggles[toggle_index];
        const int order = compare(value, toggle);
        if (order == 0) {
            ++value_index;
            ++toggle_index;
        } else if (order < 0) {
            values[output_index++] = values[value_index++];
        } else {
            values[output_index++] = toggles[toggle_index++];
        }
    }
    values.resize(new_size);
}

class BoundedHolePathExtractor {
public:
    BoundedHolePathExtractor(
        const std::unordered_map<int, std::vector<ri_meshing::BoundaryNeighborRecord>>& neighbors,
        std::size_t max_edges,
        std::vector<ri_meshing::BoundaryPath>& output
    )
        : neighbors_(neighbors), max_edges_(max_edges), output_(output)
    {
        trace_ids_.reserve(max_edges_ + 1);
    }

    void run(const std::vector<std::array<int, 2>>& seed_edges)
    {
        std::unordered_set<std::uint64_t> seen_seeds;
        seen_seeds.reserve(seed_edges.size());
        for (const std::array<int, 2> seed : seed_edges) {
            if (seed[0] < 0 || seed[1] < 0 || seed[0] == seed[1]) {
                continue;
            }
            if (!seen_seeds.insert(hole_seed_edge_key(seed)).second) {
                continue;
            }
            ++unique_seed_count_;
            if (claimed_vertices_.contains(seed[0]) && claimed_vertices_.contains(seed[1])) {
                continue;
            }
            if (trace_seed(seed)) {
                walked_vertices_ += trace_ids_.size();
                walked_steps_ += trace_ids_.size();
                claim_trace();
            }
        }
        std::sort(output_.begin(), output_.end(), boundary_path_less);
    }

    std::size_t unique_seed_count() const { return unique_seed_count_; }
    std::size_t walked_vertices() const { return walked_vertices_; }
    std::size_t walked_steps() const { return walked_steps_; }

private:
    using NeighborMap = std::unordered_map<
        int, std::vector<ri_meshing::BoundaryNeighborRecord>>;

    const ri_meshing::BoundaryNeighborRecord* find_neighbor(int vertex, int neighbor) const
    {
        const auto found = neighbors_.find(vertex);
        if (found == neighbors_.end()) {
            return nullptr;
        }
        const auto iter = std::lower_bound(
            found->second.begin(), found->second.end(), neighbor,
            [](const ri_meshing::BoundaryNeighborRecord record, int vertex_id) {
                return record.vertex < vertex_id;
            }
        );
        return iter != found->second.end() && iter->vertex == neighbor ? &*iter : nullptr;
    }

    bool trace_seed(std::array<int, 2> seed)
    {
        trace_ids_.clear();
        if (max_edges_ < 3) {
            return false;
        }
        const auto first = find_neighbor(seed[0], seed[1]);
        const auto reverse = find_neighbor(seed[1], seed[0]);
        if (first == nullptr || reverse == nullptr || !has_degree_two(seed[0])) {
            return false;
        }
        trace_ids_.push_back(seed[0]);
        int previous = seed[0];
        int current = seed[1];
        while (true) {
            if (std::find(trace_ids_.begin(), trace_ids_.end(), current) != trace_ids_.end()) {
                return false;
            }
            trace_ids_.push_back(current);
            if (trace_ids_.size() > max_edges_) {
                return false;
            }
            if (!has_degree_two(current)) {
                return false;
            }
            const auto& choices = neighbors_.at(current);
            if (choices[0].vertex != previous && choices[1].vertex != previous) {
                return false;
            }
            const int next = choices[0].vertex == previous ? choices[1].vertex : choices[0].vertex;
            if (next == start_vertex()) {
                if (trace_ids_.size() < 3) {
                    return false;
                }
                return orient_trace();
            }
            if (find_neighbor(current, next) == nullptr) {
                return false;
            }
            previous = current;
            current = next;
        }
    }

    int start_vertex() const { return trace_ids_.front(); }

    bool has_degree_two(int vertex) const
    {
        const auto found = neighbors_.find(vertex);
        return found != neighbors_.end() && found->second.size() == 2;
    }

    bool orient_trace()
    {
        int forward_votes = 0;
        int reverse_votes = 0;
        for (std::size_t index = 0; index < trace_ids_.size(); ++index) {
            const int from = trace_ids_[index];
            const int to = trace_ids_[(index + 1) % trace_ids_.size()];
            const auto record = find_neighbor(from, to);
            if (record == nullptr) {
                return false;
            }
            if (record->forward != 0) {
                ++forward_votes;
            } else {
                ++reverse_votes;
            }
        }
        if (forward_votes == reverse_votes) {
            return false;
        }
        if (reverse_votes > forward_votes) {
            std::reverse(trace_ids_.begin(), trace_ids_.end());
        }
        const auto minimum = std::min_element(trace_ids_.begin(), trace_ids_.end());
        std::rotate(trace_ids_.begin(), minimum, trace_ids_.end());
        output_.push_back(ri_meshing::BoundaryPath{trace_ids_, true});
        return true;
    }

    void claim_trace()
    {
        for (const int vertex : trace_ids_) {
            claimed_vertices_.insert(vertex);
        }
    }

    const NeighborMap& neighbors_;
    std::size_t max_edges_{};
    std::vector<ri_meshing::BoundaryPath>& output_;
    std::vector<int> trace_ids_;
    std::unordered_set<int> claimed_vertices_;
    std::size_t unique_seed_count_{};
    std::size_t walked_vertices_{};
    std::size_t walked_steps_{};
};

}  // namespace

namespace ri_meshing {

void BoundaryGraph::clear()
{
    boundary_edge_blocks_.clear();
    flattened_boundary_edges_.clear();
    flattened_boundary_edges_valid_ = false;
    boundary_edge_count_ = 0;
    sorted_boundary_vertex_ids_.clear();
    vertex_degrees_by_id_.clear();
    boundary_neighbors_.clear();
    oriented_boundary_paths_cache_.clear();
    hole_boundary_paths_cache_.clear();
    oriented_paths_by_vertex_.clear();
    oriented_path_tokens_.clear();
    oriented_path_indices_.clear();
    next_oriented_path_token_ = 1;
    dirty_boundary_vertices_.clear();
    local_boundary_marks_.clear();
    local_boundary_queue_.clear();
    local_boundary_queue_distances_.clear();
    local_boundary_edges_.clear();
    local_boundary_vertex_ids_.clear();
    local_boundary_generation_ = 1;
    oriented_path_cache_valid_ = false;
    oriented_path_cache_snapshot_ = false;
}

std::vector<int> BoundaryGraph::toggle_boundary_edges(const std::vector<Edge>& edges)
{
    if (edges.empty()) {
        return {};
    }
    oriented_path_cache_valid_ = false;
    const std::vector<Edge> toggles = sorted_odd_toggles(edges);
    if (toggles.empty()) {
        return {};
    }

    std::vector<int> touched_vertices;
    touched_vertices.reserve(toggles.size() * 2);
    std::size_t additions = 0;
    std::size_t removals = 0;
    for (const Edge toggle : toggles) {
        if (contains_boundary_edge(toggle)) {
            remove_boundary_edge(toggle);
            update_vertex_degrees(toggle, -1, touched_vertices);
            ++removals;
        } else {
            add_boundary_edge(toggle);
            update_vertex_degrees(toggle, 1, touched_vertices);
            ++additions;
        }
    }
    apply_edge_toggles(toggles);
    boundary_edge_count_ += additions - removals;
    flattened_boundary_edges_valid_ = false;
    dirty_boundary_vertices_.insert(
        dirty_boundary_vertices_.end(), touched_vertices.begin(), touched_vertices.end()
    );
    return apply_vertex_membership_changes(touched_vertices);
}

const std::vector<BoundaryGraph::Edge>& BoundaryGraph::boundary_edges() const
{
    if (!flattened_boundary_edges_valid_) {
        flattened_boundary_edges_.clear();
        flattened_boundary_edges_.reserve(boundary_edge_count_);
        for (const EdgeBlock& block : boundary_edge_blocks_) {
            flattened_boundary_edges_.insert(
                flattened_boundary_edges_.end(), block.begin(), block.end()
            );
        }
        flattened_boundary_edges_valid_ = true;
    }
    return flattened_boundary_edges_;
}

const BoundaryGraph::EdgeBlocks& BoundaryGraph::boundary_edge_blocks() const
{
    return boundary_edge_blocks_;
}

const std::vector<int>& BoundaryGraph::boundary_vertex_ids() const
{
    return sorted_boundary_vertex_ids_;
}

const std::vector<BoundaryPath>& BoundaryGraph::oriented_boundary_paths() const
{
    return oriented_boundary_paths(nullptr);
}

const std::vector<BoundaryPath>& BoundaryGraph::oriented_boundary_paths(
    BoundaryPathExtractionStats* stats
) const
{
    if (!oriented_path_cache_valid_) {
        if (oriented_path_cache_snapshot_) {
            rebuild_oriented_paths_incremental(stats);
        } else {
            rebuild_oriented_paths_full(stats);
        }
        oriented_path_cache_valid_ = true;
    }
    return oriented_boundary_paths_cache_;
}

const std::vector<BoundaryPath>& BoundaryGraph::oriented_boundary_paths_for_holes(
    const std::vector<Edge>& seed_edges,
    std::size_t max_path_edges,
    BoundaryPathExtractionStats* stats
) const
{
    hole_boundary_paths_cache_.clear();
    if (stats != nullptr) {
        *stats = BoundaryPathExtractionStats{};
    }
    if (seed_edges.empty() || boundary_neighbors_.empty()) {
        return hole_boundary_paths_cache_;
    }

    const Stopwatch walk_stopwatch;
    BoundedHolePathExtractor extractor(boundary_neighbors_, max_path_edges, hole_boundary_paths_cache_);
    extractor.run(seed_edges);
    const double walk_milliseconds = walk_stopwatch.elapsed_milliseconds();
    if (stats != nullptr) {
        stats->input_edges = seed_edges.size();
        stats->unique_edges = extractor.unique_seed_count();
        stats->undirected_paths = hole_boundary_paths_cache_.size();
        stats->oriented_paths = hole_boundary_paths_cache_.size();
        stats->dirty_seed_vertices = extractor.unique_seed_count();
        stats->dirty_collection_milliseconds = walk_milliseconds;
        stats->seeded_walk_vertices = extractor.walked_vertices();
        stats->seeded_walk_steps = extractor.walked_steps();
    }
    return hole_boundary_paths_cache_;
}

void BoundaryGraph::rebuild_oriented_path_index() const
{
    oriented_paths_by_vertex_.clear();
    oriented_path_indices_.clear();
    oriented_path_tokens_.clear();
    oriented_paths_by_vertex_.reserve(oriented_boundary_paths_cache_.size() * 4 + 1);
    oriented_path_indices_.reserve(oriented_boundary_paths_cache_.size());
    oriented_path_tokens_.reserve(oriented_boundary_paths_cache_.size());
    for (std::size_t path_index = 0; path_index < oriented_boundary_paths_cache_.size(); ++path_index) {
        const std::uint64_t token = next_oriented_path_token_++;
        oriented_path_tokens_.push_back(token);
        oriented_path_indices_.emplace(token, path_index);
        for (const int vertex : oriented_boundary_paths_cache_[path_index].ids) {
            oriented_paths_by_vertex_[vertex].push_back(token);
        }
    }
}

void BoundaryGraph::rebuild_oriented_paths_full(BoundaryPathExtractionStats* stats) const
{
    if (stats != nullptr) {
        stats->cached_paths_before = oriented_boundary_paths_cache_.size();
    }
    extract_oriented_boundary_paths_sorted_unique_into(
        boundary_edges(), sorted_boundary_vertex_ids_, stats,
        &oriented_path_workspace_, oriented_boundary_paths_cache_
    );
    rebuild_oriented_path_index();
    dirty_boundary_vertices_.clear();
    oriented_path_cache_snapshot_ = true;
    if (stats != nullptr) {
        stats->cached_paths_after = oriented_boundary_paths_cache_.size();
    }
}

void BoundaryGraph::rebuild_oriented_paths_incremental(
    BoundaryPathExtractionStats* stats
) const
{
    std::sort(dirty_boundary_vertices_.begin(), dirty_boundary_vertices_.end());
    dirty_boundary_vertices_.erase(
        std::unique(dirty_boundary_vertices_.begin(), dirty_boundary_vertices_.end()),
        dirty_boundary_vertices_.end()
    );
    const std::size_t dirty_seed_count = dirty_boundary_vertices_.size();
    const std::size_t cached_path_count_before = oriented_boundary_paths_cache_.size();
    if (dirty_boundary_vertices_.empty()) {
        if (stats != nullptr) {
            stats->cached_paths_after = oriented_boundary_paths_cache_.size();
        }
        return;
    }

    const Stopwatch collect_stopwatch;
    collect_local_boundary(
        dirty_boundary_vertices_,
        std::numeric_limits<std::size_t>::max(),
        local_boundary_edges_,
        local_boundary_vertex_ids_
    );
    const double dirty_collection_milliseconds = collect_stopwatch.elapsed_milliseconds();
    const std::size_t local_vertex_count = local_boundary_vertex_ids_.size();
    const std::size_t local_edge_count = local_boundary_edges_.size();

    std::unordered_set<std::uint64_t> affected_tokens;
    affected_tokens.reserve(dirty_boundary_vertices_.size() * 2 + 1);
    for (const int vertex : dirty_boundary_vertices_) {
        const auto found = oriented_paths_by_vertex_.find(vertex);
        if (found != oriented_paths_by_vertex_.end()) {
            affected_tokens.insert(found->second.begin(), found->second.end());
        }
    }
    const std::size_t affected_path_count_before_extraction = affected_tokens.size();

    std::vector<BoundaryPath> replacement;
    extract_oriented_boundary_paths_sorted_unique_into(
        local_boundary_edges_,
        local_boundary_vertex_ids_,
        stats,
        &oriented_path_workspace_,
        replacement
    );
    if (stats != nullptr) {
        // The path extractor resets its stats object on entry, so publish the
        // graph-level counters only after it has returned.
        stats->dirty_seed_vertices = dirty_seed_count;
        stats->cached_paths_before = cached_path_count_before;
        stats->dirty_collection_milliseconds = dirty_collection_milliseconds;
        stats->local_boundary_vertices = local_vertex_count;
        stats->local_boundary_edges = local_edge_count;
        stats->affected_path_tokens = affected_path_count_before_extraction;
        stats->replacement_paths = replacement.size();
    }

    std::size_t affected_path_count = 0;
    for (const std::uint64_t token : affected_tokens) {
        const auto path_index = oriented_path_indices_.find(token);
        if (path_index == oriented_path_indices_.end()) {
            continue;
        }
        ++affected_path_count;
        for (const int vertex : oriented_boundary_paths_cache_[path_index->second].ids) {
            const auto by_vertex = oriented_paths_by_vertex_.find(vertex);
            if (by_vertex == oriented_paths_by_vertex_.end()) {
                continue;
            }
            auto& tokens = by_vertex->second;
            tokens.erase(std::remove(tokens.begin(), tokens.end(), token), tokens.end());
            if (tokens.empty()) {
                oriented_paths_by_vertex_.erase(by_vertex);
            }
        }
    }

    struct IndexedPath {
        BoundaryPath path;
        std::uint64_t token{};
    };
    std::vector<IndexedPath> retained;
    retained.reserve(
        oriented_boundary_paths_cache_.size() - affected_path_count + replacement.size()
    );
    for (std::size_t path_index = 0; path_index < oriented_boundary_paths_cache_.size(); ++path_index) {
        const std::uint64_t token = oriented_path_tokens_[path_index];
        if (!affected_tokens.contains(token)) {
            retained.push_back({std::move(oriented_boundary_paths_cache_[path_index]), token});
        }
    }
    std::vector<IndexedPath> replacements;
    replacements.reserve(replacement.size());
    for (BoundaryPath& path : replacement) {
        const std::uint64_t token = next_oriented_path_token_++;
        for (const int vertex : path.ids) {
            oriented_paths_by_vertex_[vertex].push_back(token);
        }
        replacements.push_back({std::move(path), token});
    }
    const Stopwatch publish_stopwatch;
    std::vector<IndexedPath> updated;
    updated.reserve(retained.size() + replacements.size());
    auto less_indexed_path = [](const IndexedPath& first, const IndexedPath& second) {
        return boundary_path_less(first.path, second.path);
    };
    std::size_t retained_index = 0;
    std::size_t replacement_index = 0;
    while (retained_index < retained.size() || replacement_index < replacements.size()) {
        if (replacement_index == replacements.size()
            || (retained_index < retained.size()
                && !less_indexed_path(
                    replacements[replacement_index], retained[retained_index]
                ))) {
            updated.push_back(std::move(retained[retained_index++]));
        } else {
            updated.push_back(std::move(replacements[replacement_index++]));
        }
    }
    oriented_boundary_paths_cache_.clear();
    oriented_boundary_paths_cache_.reserve(updated.size());
    oriented_path_tokens_.clear();
    oriented_path_tokens_.reserve(updated.size());
    oriented_path_indices_.clear();
    oriented_path_indices_.reserve(updated.size());
    for (std::size_t path_index = 0; path_index < updated.size(); ++path_index) {
        oriented_boundary_paths_cache_.push_back(std::move(updated[path_index].path));
        const std::uint64_t token = updated[path_index].token;
        oriented_path_tokens_.push_back(token);
        oriented_path_indices_.emplace(token, path_index);
    }
    dirty_boundary_vertices_.clear();
    if (stats != nullptr) {
        stats->cache_publish_milliseconds = publish_stopwatch.elapsed_milliseconds();
        stats->cached_paths_after = oriented_boundary_paths_cache_.size();
    }
}

std::uint32_t BoundaryGraph::begin_local_boundary_marking() const
{
    if (local_boundary_marks_.size() < vertex_degrees_by_id_.size()) {
        local_boundary_marks_.resize(vertex_degrees_by_id_.size(), 0);
    }
    ++local_boundary_generation_;
    if (local_boundary_generation_ == 0) {
        std::fill(local_boundary_marks_.begin(), local_boundary_marks_.end(), 0);
        local_boundary_generation_ = 1;
    }
    return local_boundary_generation_;
}

void BoundaryGraph::collect_local_boundary(
    const std::vector<int>& seed_vertices,
    std::size_t max_hops,
    std::vector<Edge>& edges,
    std::vector<int>& vertex_ids
) const
{
    edges.clear();
    vertex_ids.clear();
    if (seed_vertices.empty() || boundary_neighbors_.empty()) {
        return;
    }

    const std::uint32_t generation = begin_local_boundary_marking();
    local_boundary_queue_.clear();
    local_boundary_queue_.reserve(seed_vertices.size() * 4 + 1);
    const bool bounded = max_hops != std::numeric_limits<std::size_t>::max();
    local_boundary_queue_distances_.clear();
    if (bounded) {
        local_boundary_queue_distances_.reserve(seed_vertices.size() * 4 + 1);
    }
    for (const int seed : seed_vertices) {
        if (seed < 0 || static_cast<std::size_t>(seed) >= local_boundary_marks_.size()) {
            continue;
        }
        if (local_boundary_marks_[static_cast<std::size_t>(seed)] != generation) {
            local_boundary_marks_[static_cast<std::size_t>(seed)] = generation;
            local_boundary_queue_.push_back(seed);
            if (bounded) {
                local_boundary_queue_distances_.push_back(0);
            }
        }
    }
    for (std::size_t queue_index = 0; queue_index < local_boundary_queue_.size(); ++queue_index) {
        const int vertex = local_boundary_queue_[queue_index];
        const std::size_t distance = bounded ? local_boundary_queue_distances_[queue_index] : 0;
        if (distance >= max_hops) {
            continue;
        }
        const auto neighbors = boundary_neighbors_.find(vertex);
        if (neighbors == boundary_neighbors_.end()) {
            continue;
        }
        for (const BoundaryNeighborRecord neighbor : neighbors->second) {
            if (neighbor.vertex < 0
                || static_cast<std::size_t>(neighbor.vertex) >= local_boundary_marks_.size()
                || local_boundary_marks_[static_cast<std::size_t>(neighbor.vertex)] == generation) {
                continue;
            }
            local_boundary_marks_[static_cast<std::size_t>(neighbor.vertex)] = generation;
            local_boundary_queue_.push_back(neighbor.vertex);
            if (bounded) {
                local_boundary_queue_distances_.push_back(distance + 1);
            }
        }
    }

    vertex_ids = local_boundary_queue_;
    std::sort(vertex_ids.begin(), vertex_ids.end());

    for (const int vertex : vertex_ids) {
        const auto neighbors = boundary_neighbors_.find(vertex);
        if (neighbors == boundary_neighbors_.end()) {
            continue;
        }
        for (const BoundaryNeighborRecord neighbor : neighbors->second) {
            if (vertex < neighbor.vertex
                && neighbor.vertex >= 0
                && static_cast<std::size_t>(neighbor.vertex) < local_boundary_marks_.size()
                && local_boundary_marks_[static_cast<std::size_t>(neighbor.vertex)] == generation) {
                edges.push_back(neighbor.forward
                    ? Edge{vertex, neighbor.vertex}
                    : Edge{neighbor.vertex, vertex});
            }
        }
    }
}

std::size_t BoundaryGraph::boundary_edge_count() const
{
    return boundary_edge_count_;
}

std::array<std::size_t, 4> BoundaryGraph::incidence_histogram() const
{
    return {boundary_edge_count_, 0, 0, 0};
}

std::uint64_t BoundaryGraph::edge_key(int a, int b)
{
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

bool BoundaryGraph::edge_less(Edge a, Edge b)
{
    const std::uint64_t key_a = edge_key(a[0], a[1]);
    const std::uint64_t key_b = edge_key(b[0], b[1]);
    if (key_a != key_b) {
        return key_a < key_b;
    }
    return a < b;
}

std::vector<BoundaryGraph::Edge> BoundaryGraph::sorted_odd_toggles(
    const std::vector<Edge>& edges
)
{
    std::vector<Edge> sorted = edges;
    std::sort(sorted.begin(), sorted.end(), edge_less);

    std::vector<Edge> out;
    out.reserve(sorted.size());
    for (std::size_t begin = 0; begin < sorted.size();) {
        std::size_t end = begin + 1;
        const std::uint64_t key = edge_key(sorted[begin][0], sorted[begin][1]);
        while (end < sorted.size() && edge_key(sorted[end][0], sorted[end][1]) == key) {
            ++end;
        }
        if (((end - begin) & 1U) != 0U) {
            out.push_back(sorted[begin]);
        }
        begin = end;
    }
    return out;
}

void BoundaryGraph::add_neighbor(
    std::unordered_map<int, std::vector<BoundaryNeighborRecord>>& neighbors,
    int vertex,
    int neighbor,
    bool forward
)
{
    auto& values = neighbors[vertex];
    const auto position = std::lower_bound(
        values.begin(), values.end(), neighbor,
        [](const BoundaryNeighborRecord item, int vertex_id) {
            return item.vertex < vertex_id;
        }
    );
    values.insert(
        position,
        BoundaryNeighborRecord{neighbor, static_cast<std::uint8_t>(forward ? 1 : 0)}
    );
}

void BoundaryGraph::remove_neighbor(
    std::unordered_map<int, std::vector<BoundaryNeighborRecord>>& neighbors,
    int vertex,
    int neighbor
)
{
    const auto found = neighbors.find(vertex);
    if (found == neighbors.end()) {
        return;
    }
    auto& values = found->second;
    const auto iter = std::lower_bound(
        values.begin(), values.end(), neighbor,
        [](const BoundaryNeighborRecord item, int vertex_id) {
            return item.vertex < vertex_id;
        }
    );
    if (iter != values.end() && iter->vertex == neighbor) {
        values.erase(iter);
    }
    if (values.empty()) {
        neighbors.erase(found);
    }
}

void BoundaryGraph::add_boundary_edge(Edge edge)
{
    if (edge[0] < 0 || edge[1] < 0 || edge[0] == edge[1]) {
        return;
    }
    add_neighbor(boundary_neighbors_, edge[0], edge[1], true);
    add_neighbor(boundary_neighbors_, edge[1], edge[0], false);
}

void BoundaryGraph::remove_boundary_edge(Edge edge)
{
    if (edge[0] < 0 || edge[1] < 0 || edge[0] == edge[1]) {
        return;
    }
    remove_neighbor(boundary_neighbors_, edge[0], edge[1]);
    remove_neighbor(boundary_neighbors_, edge[1], edge[0]);
}

void BoundaryGraph::update_vertex_degrees(Edge edge, int delta, std::vector<int>& touched_vertices)
{
    for (const int vertex_id : edge) {
        if (vertex_id < 0) {
            continue;
        }
        if (static_cast<std::size_t>(vertex_id) >= vertex_degrees_by_id_.size()) {
            vertex_degrees_by_id_.resize(static_cast<std::size_t>(vertex_id) + 1, 0);
        }
        vertex_degrees_by_id_[static_cast<std::size_t>(vertex_id)] += delta;
        touched_vertices.push_back(vertex_id);
    }
}

bool BoundaryGraph::contains_boundary_edge(Edge edge) const
{
    if (boundary_edge_blocks_.empty()) {
        return false;
    }
    const std::uint64_t target_key = edge_key(edge[0], edge[1]);
    const auto block_iter = std::lower_bound(
        boundary_edge_blocks_.begin(), boundary_edge_blocks_.end(), target_key,
        [](const EdgeBlock& block, std::uint64_t key) {
            return edge_key(block.back()[0], block.back()[1]) < key;
        }
    );
    if (block_iter == boundary_edge_blocks_.end()) {
        return false;
    }
    const auto edge_iter = std::lower_bound(
        block_iter->begin(), block_iter->end(), target_key,
        [](const Edge first, std::uint64_t second) {
            return edge_key(first[0], first[1]) < second;
        }
    );
    return edge_iter != block_iter->end()
        && edge_key((*edge_iter)[0], (*edge_iter)[1]) == target_key;
}

void BoundaryGraph::apply_edge_toggles(const std::vector<Edge>& toggles)
{
    EdgeBlocks updated;
    updated.reserve(
        boundary_edge_blocks_.size() + toggles.size() / edge_block_target_size_ + 1
    );
    std::size_t toggle_index = 0;
    for (std::size_t block_index = 0; block_index < boundary_edge_blocks_.size(); ++block_index) {
        EdgeBlock block = std::move(boundary_edge_blocks_[block_index]);
        const std::uint64_t upper_key = block_index + 1 < boundary_edge_blocks_.size()
            ? edge_key(
                boundary_edge_blocks_[block_index + 1].front()[0],
                boundary_edge_blocks_[block_index + 1].front()[1]
            )
            : std::numeric_limits<std::uint64_t>::max();
        const std::size_t first_toggle = toggle_index;
        while (toggle_index < toggles.size()
            && edge_key(toggles[toggle_index][0], toggles[toggle_index][1]) < upper_key) {
            ++toggle_index;
        }
        if (first_toggle == toggle_index) {
            updated.push_back(std::move(block));
            continue;
        }
        EdgeBlock merged;
        merged.reserve(block.size() + toggle_index - first_toggle);
        std::size_t value_index = 0;
        std::size_t current_toggle = first_toggle;
        while (value_index < block.size() || current_toggle < toggle_index) {
            if (value_index == block.size()) {
                merged.push_back(toggles[current_toggle++]);
                continue;
            }
            if (current_toggle == toggle_index) {
                merged.push_back(block[value_index++]);
                continue;
            }
            const int order = compare_sorted_edges(block[value_index], toggles[current_toggle]);
            if (order == 0) {
                ++value_index;
                ++current_toggle;
            } else if (order < 0) {
                merged.push_back(block[value_index++]);
            } else {
                merged.push_back(toggles[current_toggle++]);
            }
        }
        for (std::size_t begin = 0; begin < merged.size(); begin += edge_block_target_size_) {
            const std::size_t end = std::min(begin + edge_block_target_size_, merged.size());
            EdgeBlock chunk;
            chunk.reserve(end - begin);
            chunk.insert(chunk.end(), merged.begin() + begin, merged.begin() + end);
            updated.push_back(std::move(chunk));
        }
    }
    while (toggle_index < toggles.size()) {
        EdgeBlock chunk;
        const std::size_t end = std::min(toggle_index + edge_block_target_size_, toggles.size());
        chunk.insert(chunk.end(), toggles.begin() + toggle_index, toggles.begin() + end);
        updated.push_back(std::move(chunk));
        toggle_index = end;
    }
    boundary_edge_blocks_.swap(updated);
}

std::vector<int> BoundaryGraph::apply_vertex_membership_changes(std::vector<int>& touched_vertices)
{
    if (touched_vertices.empty()) {
        return {};
    }
    std::sort(touched_vertices.begin(), touched_vertices.end());
    touched_vertices.erase(std::unique(touched_vertices.begin(), touched_vertices.end()), touched_vertices.end());

    std::vector<int> toggles;
    toggles.reserve(touched_vertices.size());
    std::size_t additions = 0;
    std::size_t removals = 0;
    for (const int vertex_id : touched_vertices) {
        const bool was_present = std::binary_search(
            sorted_boundary_vertex_ids_.begin(), sorted_boundary_vertex_ids_.end(), vertex_id
        );
        const bool is_present = vertex_id >= 0
            && static_cast<std::size_t>(vertex_id) < vertex_degrees_by_id_.size()
            && vertex_degrees_by_id_[static_cast<std::size_t>(vertex_id)] > 0;
        if (was_present != is_present) {
            toggles.push_back(vertex_id);
            if (is_present) {
                ++additions;
            } else {
                ++removals;
            }
        }
    }
    if (toggles.empty()) {
        return {};
    }

    merge_sorted_toggles_in_place(
        sorted_boundary_vertex_ids_, toggles, additions, removals, compare_sorted_ids
    );
    return toggles;
}

}  // namespace ri_meshing
