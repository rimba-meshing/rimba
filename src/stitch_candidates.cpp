#include "stitch_core_components.hpp"
#include "stitch_boundary_filters.hpp"

#include "ri_meshing/timing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if defined(RI_MESHING_HAS_OPENMP)
#include <omp.h>
#endif

namespace ri_meshing::stitch_detail {
namespace {

struct NewPathWork {
    std::vector<int> ids;
    std::vector<MeshScalar> labels;
    std::vector<Edge> edges;
    std::unordered_map<int, std::vector<int>> neighbors;
    bool closed{};
};

struct PreparedStitchPaths {
    OldLabelSegment old_segment;
    NewPathWork new_work;
    std::vector<std::array<std::size_t, 2>> collapsed_old_intervals;
};

struct CandidateTask {
    int cluster{};
    int segment{};
};

struct PreparedSegmentWork {
    int cluster{};
    bool local_bridge{};
    std::vector<int> replaced_new_vertices;
    PreparedStitchPaths paths;
};

struct BoundaryHalfEdge {
    int path{-1};
    int junction{-1};
    int neighbor{-1};
};

struct OldJunction {
    int vertex{-1};
    std::vector<BoundaryHalfEdge> branches;
};

struct JunctionSegmentSupport {
    int segment{-1};
    int junction{-1};
    std::vector<BoundaryHalfEdge> branches;
};

using JunctionSupportMap = std::vector<std::unordered_map<
    int, std::vector<JunctionSegmentSupport>>>;

constexpr MeshScalar kTiedLabelEpsilon = static_cast<MeshScalar>(1e-9);
constexpr MeshScalar kSyntheticTieOffset = static_cast<MeshScalar>(1e-7);

std::vector<std::array<std::size_t, 2>> collapsed_old_intervals(
    const OldLabelSegment& segment
)
{
    std::vector<std::array<std::size_t, 2>> intervals;
    if (segment.ids.size() != segment.labels.size()) {
        return intervals;
    }
    for (std::size_t index = 0; index + 1 < segment.labels.size(); ++index) {
        if (std::abs(segment.labels[index + 1] - segment.labels[index])
            <= kTiedLabelEpsilon) {
            intervals.push_back({index, index + 1});
        }
    }
    return intervals;
}

bool has_tied_labels(const OldLabelSegment& segment)
{
    if (segment.labels.size() < 2) {
        return false;
    }
    const auto [minimum, maximum] = std::minmax_element(
        segment.labels.begin(), segment.labels.end()
    );
    return *maximum - *minimum <= kTiedLabelEpsilon;
}

Vec3f point_delta(Vec3f first, Vec3f second)
{
    return Vec3f{second.x - first.x, second.y - first.y, second.z - first.z};
}

MeshScalar squared_norm(Vec3f value)
{
    return static_cast<MeshScalar>(value.x) * value.x
        + static_cast<MeshScalar>(value.y) * value.y
        + static_cast<MeshScalar>(value.z) * value.z;
}

MeshScalar dot_product(Vec3f first, Vec3f second)
{
    return static_cast<MeshScalar>(first.x) * second.x
        + static_cast<MeshScalar>(first.y) * second.y
        + static_cast<MeshScalar>(first.z) * second.z;
}

std::size_t nearest_path_position(
    const BoundaryPath& path,
    MeshScalar target_label
)
{
    if (path.ids.size() < 2) {
        return 0;
    }
    const MeshScalar denominator = static_cast<MeshScalar>(path.ids.size() - 1);
    const MeshScalar period = path.closed
        ? static_cast<MeshScalar>(path.ids.size()) / denominator : static_cast<MeshScalar>(1);
    std::size_t best = 0;
    MeshScalar best_distance = std::numeric_limits<MeshScalar>::infinity();
    for (std::size_t index = 0; index < path.ids.size(); ++index) {
        const MeshScalar base = static_cast<MeshScalar>(index) / denominator;
        for (int cycle = -2; cycle <= 2; ++cycle) {
            const MeshScalar distance = std::abs(base + cycle * period - target_label);
            if (distance < best_distance) {
                best = index;
                best_distance = distance;
            }
        }
    }
    return best;
}

std::size_t advance_path_position(
    const BoundaryPath& path,
    std::size_t start,
    std::size_t steps
)
{
    if (path.ids.empty()) {
        return 0;
    }
    if (path.closed) {
        return (start + steps) % path.ids.size();
    }
    return std::min(start + steps, path.ids.size() - 1);
}

bool should_reverse_tied_segment(
    const OldLabelSegment& old_segment,
    const BoundaryPath& new_path,
    const std::vector<Vec3f>& vertices
)
{
    if (!has_tied_labels(old_segment) || old_segment.ids.size() < 2
        || new_path.ids.size() < 2) {
        return false;
    }
    const int old_first = old_segment.ids.front();
    const int old_last = old_segment.ids.back();
    if (old_first < 0 || old_last < 0
        || static_cast<std::size_t>(old_first) >= vertices.size()
        || static_cast<std::size_t>(old_last) >= vertices.size()) {
        return false;
    }

    const std::size_t new_start = nearest_path_position(
        new_path, old_segment.labels.front()
    );
    const std::size_t new_span = std::min(
        old_segment.ids.size() - 1,
        new_path.ids.size() - 1
    );
    const std::size_t new_end = advance_path_position(new_path, new_start, new_span);
    const int new_first = new_path.ids[new_start];
    const int new_last = new_path.ids[new_end];
    if (new_first < 0 || new_last < 0
        || static_cast<std::size_t>(new_first) >= vertices.size()
        || static_cast<std::size_t>(new_last) >= vertices.size()) {
        return false;
    }

    const Vec3f old_direction = point_delta(
        vertices[static_cast<std::size_t>(old_first)],
        vertices[static_cast<std::size_t>(old_last)]
    );
    const Vec3f new_direction = point_delta(
        vertices[static_cast<std::size_t>(new_first)],
        vertices[static_cast<std::size_t>(new_last)]
    );
    const MeshScalar old_length_sq = squared_norm(old_direction);
    const MeshScalar new_length_sq = squared_norm(new_direction);
    if (old_length_sq <= static_cast<MeshScalar>(1e-12)
        || new_length_sq <= static_cast<MeshScalar>(1e-12)) {
        return false;
    }
    const MeshScalar cosine = dot_product(old_direction, new_direction)
        / std::sqrt(old_length_sq * new_length_sq);
    if (cosine < static_cast<MeshScalar>(-0.05)) {
        return true;
    }
    if (cosine > static_cast<MeshScalar>(0.05)) {
        return false;
    }

    const Vec3f& old_start_point = vertices[static_cast<std::size_t>(old_first)];
    const Vec3f& old_end_point = vertices[static_cast<std::size_t>(old_last)];
    const Vec3f& new_start_point = vertices[static_cast<std::size_t>(new_first)];
    const Vec3f& new_end_point = vertices[static_cast<std::size_t>(new_last)];
    const MeshScalar forward_cost = squared_norm(point_delta(old_start_point, new_start_point))
        + squared_norm(point_delta(old_end_point, new_end_point));
    const MeshScalar reverse_cost = squared_norm(point_delta(old_start_point, new_end_point))
        + squared_norm(point_delta(old_end_point, new_start_point));
    return reverse_cost + static_cast<MeshScalar>(1e-12) < forward_cost;
}

bool path_contains_edge(const BoundaryPath& path, int first, int second)
{
    if (path.ids.size() < 2) {
        return false;
    }
    for (std::size_t index = 0; index + 1 < path.ids.size(); ++index) {
        if ((path.ids[index] == first && path.ids[index + 1] == second)
            || (path.ids[index] == second && path.ids[index + 1] == first)) {
            return true;
        }
    }
    return path.closed
        && ((path.ids.back() == first && path.ids.front() == second)
            || (path.ids.back() == second && path.ids.front() == first));
}

bool same_half_edge(const BoundaryHalfEdge& first, const BoundaryHalfEdge& second)
{
    return first.path == second.path && first.neighbor == second.neighbor;
}

void add_unique_half_edge(
    std::vector<BoundaryHalfEdge>& edges,
    BoundaryHalfEdge edge
)
{
    if (std::find_if(
            edges.begin(), edges.end(),
            [&](const BoundaryHalfEdge& existing) { return same_half_edge(existing, edge); }
        ) == edges.end()) {
        edges.push_back(edge);
    }
}

std::vector<OldJunction> build_old_junction_incidence_graph(
    const std::vector<BoundaryPath>& old_paths
)
{
    struct Accumulator {
        int occurrences{};
        std::unordered_set<int> paths;
        std::vector<BoundaryHalfEdge> branches;
    };
    std::unordered_map<int, Accumulator> accumulated;
    for (std::size_t path_index = 0; path_index < old_paths.size(); ++path_index) {
        const BoundaryPath& path = old_paths[path_index];
        for (std::size_t position = 0; position < path.ids.size(); ++position) {
            const int vertex = path.ids[position];
            Accumulator& item = accumulated[vertex];
            ++item.occurrences;
            item.paths.insert(static_cast<int>(path_index));
            if (position > 0) {
                add_unique_half_edge(item.branches, BoundaryHalfEdge{
                    static_cast<int>(path_index), vertex, path.ids[position - 1]
                });
            } else if (path.closed && path.ids.size() > 1) {
                add_unique_half_edge(item.branches, BoundaryHalfEdge{
                    static_cast<int>(path_index), vertex, path.ids.back()
                });
            }
            if (position + 1 < path.ids.size()) {
                add_unique_half_edge(item.branches, BoundaryHalfEdge{
                    static_cast<int>(path_index), vertex, path.ids[position + 1]
                });
            } else if (path.closed && path.ids.size() > 1) {
                add_unique_half_edge(item.branches, BoundaryHalfEdge{
                    static_cast<int>(path_index), vertex, path.ids.front()
                });
            }
        }
    }

    std::vector<OldJunction> junctions;
    for (const auto& [vertex, item] : accumulated) {
        const bool shared_path = item.paths.size() > 1;
        const bool repeated_on_path = item.occurrences > 1;
        if (!(shared_path || repeated_on_path) || item.branches.size() < 2) {
            continue;
        }
        junctions.push_back(OldJunction{vertex, item.branches});
    }
    return junctions;
}

int nearest_path_vertex(const BoundaryPath& path, MeshScalar target)
{
    if (path.ids.empty()) {
        return -1;
    }
    const MeshScalar denominator = path.ids.size() > 1
        ? static_cast<MeshScalar>(path.ids.size() - 1) : static_cast<MeshScalar>(1);
    int best_vertex = path.ids.front();
    MeshScalar best_distance = std::numeric_limits<MeshScalar>::infinity();
    for (std::size_t index = 0; index < path.ids.size(); ++index) {
        const MeshScalar value = static_cast<MeshScalar>(index) / denominator;
        MeshScalar distance = std::abs(value - target);
        if (path.closed) {
            distance = std::min(distance, static_cast<MeshScalar>(1) - distance);
        }
        if (distance < best_distance) {
            best_distance = distance;
            best_vertex = path.ids[index];
        }
    }
    return best_vertex;
}
std::uint64_t edge_key(int a, int b)
{
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

std::array<int, 3> face_key(Triangle face)
{
    std::array<int, 3> ids{face.v0, face.v1, face.v2};
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool face_uses_edge(Triangle face, int first, int second)
{
    const Edge edges[3]{{face.v0, face.v1}, {face.v1, face.v2}, {face.v2, face.v0}};
    return std::any_of(
        std::begin(edges), std::end(edges),
        [&](const Edge edge) {
            return (edge[0] == first && edge[1] == second)
                || (edge[0] == second && edge[1] == first);
        }
    );
}
std::vector<std::array<int, 2>> greedy_connecting_edge_indices(
    const std::vector<MeshScalar>& old_labels,
    const std::vector<MeshScalar>& new_labels
)
{
    if (!std::is_sorted(old_labels.begin(), old_labels.end())) {
        struct Item {
            MeshScalar label;
            int source;
            int index;
        };
        std::vector<Item> items;
        items.reserve(old_labels.size() + new_labels.size());
        for (std::size_t idx = 0; idx < old_labels.size(); ++idx) {
            items.push_back({old_labels[idx], 0, static_cast<int>(idx)});
        }
        for (std::size_t idx = 0; idx < new_labels.size(); ++idx) {
            items.push_back({new_labels[idx], 1, static_cast<int>(idx)});
        }
        std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            return a.label < b.label;
        });

        std::array<int, 2> current{-1, -1};
        std::vector<std::array<int, 2>> edges;
        edges.reserve(items.size());
        for (const Item item : items) {
            current[static_cast<std::size_t>(item.source)] = item.index;
            if (current[0] < 0 || current[1] < 0) {
                continue;
            }
            edges.push_back(current);
        }
        return edges;
    }

    std::array<int, 2> current{-1, -1};
    std::vector<std::array<int, 2>> edges;
    edges.reserve(old_labels.size() + new_labels.size());
    std::size_t old_idx = 0;
    std::size_t new_idx = 0;
    while (old_idx < old_labels.size() || new_idx < new_labels.size()) {
        if (new_idx >= new_labels.size()
            || (old_idx < old_labels.size() && old_labels[old_idx] <= new_labels[new_idx])) {
            current[0] = static_cast<int>(old_idx);
            ++old_idx;
        } else {
            current[1] = static_cast<int>(new_idx);
            ++new_idx;
        }
        if (current[0] < 0 || current[1] < 0) {
            continue;
        }
        edges.push_back(current);
    }
    return edges;
}

std::uint64_t compact_face_key(std::array<int, 3> key)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key[0])) << 42U)
        ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key[1])) << 21U)
        ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(key[2]));
}

std::uint64_t compact_stitch_vertex_key(int old_id, int new_id)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(old_id)) << 32U)
        | static_cast<std::uint64_t>(static_cast<std::uint32_t>(new_id));
}

void add_neighbor(std::unordered_map<int, std::vector<int>>& neighbors, int a, int b)
{
    std::vector<int>& a_neighbors = neighbors[a];
    if (std::find(a_neighbors.begin(), a_neighbors.end(), b) == a_neighbors.end()) {
        a_neighbors.push_back(b);
    }
    std::vector<int>& b_neighbors = neighbors[b];
    if (std::find(b_neighbors.begin(), b_neighbors.end(), a) == b_neighbors.end()) {
        b_neighbors.push_back(a);
    }
}

std::vector<Edge> path_edges(const std::vector<int>& ids, bool closed)
{
    std::vector<Edge> edges;
    if (ids.size() < 2) {
        return edges;
    }
    edges.reserve(ids.size());
    for (std::size_t idx = 0; idx + 1 < ids.size(); ++idx) {
        edges.push_back({ids[idx], ids[idx + 1]});
    }
    if (closed) {
        edges.push_back({ids.back(), ids.front()});
    }
    return edges;
}

int compare_direction_votes(const std::vector<MeshScalar>& labels)
{
    int votes = 0;
    for (std::size_t idx = 1; idx < labels.size(); ++idx) {
        if (labels[idx] > labels[idx - 1]) {
            ++votes;
        } else if (labels[idx] < labels[idx - 1]) {
            --votes;
        }
    }
    return votes;
}

std::vector<MeshScalar> unwrap_cyclic_labels(
    const std::vector<MeshScalar>& labels,
    MeshScalar period
)
{
    std::vector<MeshScalar> out(labels.size());
    if (labels.empty()) {
        return out;
    }
    out[0] = labels[0];
    MeshScalar offset = static_cast<MeshScalar>(0);
    MeshScalar previous = out[0];
    const MeshScalar half_period = static_cast<MeshScalar>(0.5) * period;
    for (std::size_t idx = 1; idx < labels.size(); ++idx) {
        MeshScalar value = labels[idx] + offset;
        while (value - previous > half_period) {
            offset -= period;
            value -= period;
        }
        while (value - previous < -half_period) {
            offset += period;
            value += period;
        }
        out[idx] = value;
        previous = value;
    }
    return out;
}

void roll_to_minimum_label(OldLabelSegment& segment)
{
    if (segment.labels.empty()) {
        return;
    }
    const auto min_iter = std::min_element(segment.labels.begin(), segment.labels.end());
    const auto offset = static_cast<std::ptrdiff_t>(min_iter - segment.labels.begin());
    std::rotate(segment.ids.begin(), segment.ids.begin() + offset, segment.ids.end());
    std::rotate(segment.labels.begin(), segment.labels.begin() + offset, segment.labels.end());
}

void make_labels_strictly_increasing(
    std::vector<MeshScalar>& labels,
    MeshScalar fallback_step
)
{
    if (labels.size() <= 1) {
        return;
    }

    for (std::size_t idx = 1; idx < labels.size(); ++idx) {
        labels[idx] = std::max(labels[idx], labels[idx - 1]);
    }

    std::size_t start = 0;
    while (start < labels.size()) {
        std::size_t end = start + 1;
        while (end < labels.size() && labels[end] == labels[start]) {
            ++end;
        }

        const std::size_t run_len = end - start;
        if (run_len > 1) {
            if (end < labels.size()) {
                const MeshScalar first = labels[start];
                const MeshScalar next = labels[end];
                const MeshScalar step = (next - first) / static_cast<MeshScalar>(run_len);
                for (std::size_t idx = 0; idx < run_len; ++idx) {
                    labels[start + idx] = first + step * static_cast<MeshScalar>(idx);
                }
            } else if (start > 0) {
                const MeshScalar previous = labels[start - 1];
                const MeshScalar last = labels[start];
                const MeshScalar step = (last - previous) / static_cast<MeshScalar>(run_len);
                for (std::size_t idx = 0; idx < run_len; ++idx) {
                    labels[start + idx] = previous + step * static_cast<MeshScalar>(idx + 1);
                }
            } else {
                // A completely tied segment has no measured span. Keep its
                // synthetic ordering at the scale of one new-path sample;
                // expanding it by whole label units can make a short partial
                // segment appear to cover an entire closed contour.
                for (std::size_t idx = 0; idx < run_len; ++idx) {
                    labels[start + idx] = labels[start]
                        + fallback_step * static_cast<MeshScalar>(idx);
                }
            }
        }

        start = end;
    }
}

void bias_tied_segment_labels(std::vector<MeshScalar>& labels)
{
    for (MeshScalar& label : labels) {
        label += kSyntheticTieOffset;
    }
}

NewPathWork make_new_path_work_from_arc(
    std::vector<int> ids,
    std::vector<MeshScalar> labels,
    bool closed
)
{
    NewPathWork work;
    if (ids.size() < 2 || ids.size() != labels.size()) {
        return work;
    }
    work.ids = std::move(ids);
    work.labels = std::move(labels);
    work.closed = closed;
    work.edges = path_edges(work.ids, closed);
    work.neighbors.reserve(work.ids.size());
    for (const Edge edge : work.edges) {
        add_neighbor(work.neighbors, edge[0], edge[1]);
    }
    return work;
}

NewPathWork extract_cyclic_new_arc(
    const BoundaryPath& new_path,
    MeshScalar arc_min,
    MeshScalar arc_max
)
{
    if (new_path.ids.size() < 2 || arc_max < arc_min) {
        return {};
    }

    const MeshScalar denom = static_cast<MeshScalar>(new_path.ids.size() - 1);
    const MeshScalar period = static_cast<MeshScalar>(new_path.ids.size()) / denom;
    const MeshScalar include_min = arc_min - (static_cast<MeshScalar>(0.5) / denom);
    const MeshScalar include_max = arc_max + (static_cast<MeshScalar>(0.5) / denom);
    std::vector<std::pair<MeshScalar, int>> candidates;
    candidates.reserve(new_path.ids.size());
    for (std::size_t idx = 0; idx < new_path.ids.size(); ++idx) {
        const MeshScalar base_label = static_cast<MeshScalar>(idx) / denom;
        int min_cycle = static_cast<int>(std::floor((include_min - base_label) / period)) - 1;
        int max_cycle = static_cast<int>(std::ceil((include_max - base_label) / period)) + 1;
        for (int cycle = min_cycle; cycle <= max_cycle; ++cycle) {
            const MeshScalar label = base_label + static_cast<MeshScalar>(cycle) * period;
            if (label >= include_min && label <= include_max) {
                candidates.push_back({label, new_path.ids[idx]});
                break;
            }
        }
    }
    if (candidates.size() < 2) {
        return {};
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    std::vector<int> ids;
    std::vector<MeshScalar> labels;
    ids.reserve(candidates.size());
    labels.reserve(candidates.size());
    for (const auto [label, vertex_id] : candidates) {
        ids.push_back(vertex_id);
        labels.push_back(label);
    }

    const bool covers_full_cycle = (arc_max - arc_min)
        >= period - (static_cast<MeshScalar>(1.5) / denom);
    return make_new_path_work_from_arc(std::move(ids), std::move(labels), covers_full_cycle);
}

MeshScalar label_after(MeshScalar base, MeshScalar reference, MeshScalar period)
{
    const MeshScalar cycles = std::floor((reference - base) / period);
    MeshScalar value = base + cycles * period;
    if (value <= reference) {
        value += period;
    }
    return value;
}

NewPathWork restore_seam_samples(
    const NewPathWork& full_work,
    const NewPathWork& arc_work
)
{
    if (full_work.ids.size() < 3 || arc_work.ids.size() >= full_work.ids.size()) {
        return {};
    }
    const std::size_t missing_count = full_work.ids.size() - arc_work.ids.size();
    constexpr std::size_t kMaxRestoredSeamSamples = 4;
    if (missing_count > kMaxRestoredSeamSamples) {
        return {};
    }

    std::vector<bool> present(full_work.ids.size(), false);
    for (const int vertex : arc_work.ids) {
        const auto match = std::find(full_work.ids.begin(), full_work.ids.end(), vertex);
        if (match == full_work.ids.end()) {
            return {};
        }
        const std::size_t index = static_cast<std::size_t>(match - full_work.ids.begin());
        if (present[index]) {
            return {};
        }
        present[index] = true;
    }
    std::vector<std::size_t> missing_indices;
    missing_indices.reserve(missing_count);
    for (std::size_t index = 0; index < present.size(); ++index) {
        if (!present[index]) {
            missing_indices.push_back(index);
        }
    }
    if (missing_indices.size() != missing_count) {
        return {};
    }

    const MeshScalar period = static_cast<MeshScalar>(full_work.ids.size())
        / static_cast<MeshScalar>(full_work.ids.size() - 1);

    // The arc is sorted in unwrapped label order, so a seam omission appears
    // between its last and first samples. Accept exactly one cyclic missing
    // run and verify that the arc contains the complementary path interval.
    std::size_t present_to_missing = 0;
    std::size_t missing_start = 0;
    for (std::size_t index = 0; index < present.size(); ++index) {
        const std::size_t next = (index + 1) % present.size();
        if (present[index] && !present[next]) {
            ++present_to_missing;
            missing_start = next;
        }
    }
    if (present_to_missing != 1) {
        return {};
    }

    std::vector<std::size_t> missing_cyclic;
    missing_cyclic.reserve(missing_count);
    std::size_t index = missing_start;
    while (!present[index]) {
        missing_cyclic.push_back(index);
        index = (index + 1) % present.size();
    }
    if (missing_cyclic.size() != missing_count) {
        return {};
    }

    const std::size_t first_present = (missing_start + missing_count) % present.size();
    std::vector<int> expected_ids;
    expected_ids.reserve(arc_work.ids.size());
    for (std::size_t offset = 0; offset < arc_work.ids.size(); ++offset) {
        expected_ids.push_back(full_work.ids[(first_present + offset) % full_work.ids.size()]);
    }
    if (expected_ids != arc_work.ids) {
        return {};
    }

    std::vector<int> ids = arc_work.ids;
    std::vector<MeshScalar> labels = arc_work.labels;
    for (const std::size_t missing_index : missing_cyclic) {
        ids.push_back(full_work.ids[missing_index]);
        labels.push_back(label_after(full_work.labels[missing_index], labels.back(), period));
    }
    return make_new_path_work_from_arc(std::move(ids), std::move(labels), true);
}

PreparedStitchPaths prepare_stitch_paths(
    const BoundaryPath& new_path,
    const NewPathWork& new_work,
    OldLabelSegment old_segment,
    const std::vector<Vec3f>& vertices
)
{
    if (!new_path.closed || new_path.ids.size() < 3) {
        const bool tied = has_tied_labels(old_segment);
        if (compare_direction_votes(old_segment.labels) < 0
            || should_reverse_tied_segment(old_segment, new_path, vertices)) {
            std::reverse(old_segment.ids.begin(), old_segment.ids.end());
            std::reverse(old_segment.labels.begin(), old_segment.labels.end());
        }
        const std::vector<std::array<std::size_t, 2>> collapsed =
            collapsed_old_intervals(old_segment);
        make_labels_strictly_increasing(old_segment.labels, static_cast<MeshScalar>(1));
        if (tied) {
            bias_tied_segment_labels(old_segment.labels);
        }
        return PreparedStitchPaths{
            std::move(old_segment), new_work, collapsed
        };
    }

    const MeshScalar denom = static_cast<MeshScalar>(new_path.ids.size() - 1);
    const MeshScalar period = static_cast<MeshScalar>(new_path.ids.size()) / denom;
    old_segment.labels = unwrap_cyclic_labels(old_segment.labels, period);
    const bool tied = has_tied_labels(old_segment);
    if (compare_direction_votes(old_segment.labels) < 0
        || should_reverse_tied_segment(old_segment, new_path, vertices)) {
        std::reverse(old_segment.ids.begin(), old_segment.ids.end());
        std::reverse(old_segment.labels.begin(), old_segment.labels.end());
        old_segment.labels = unwrap_cyclic_labels(old_segment.labels, period);
    }
    // Rolling is a cyclic operation.  It is valid for a complete old loop,
    // but an open old segment has physical endpoints that must remain fixed;
    // rotating it can make one endpoint appear next to the opposite side of
    // the closed new contour.
    if (old_segment.closed) {
        roll_to_minimum_label(old_segment);
    }
    const std::vector<std::array<std::size_t, 2>> collapsed =
        collapsed_old_intervals(old_segment);
    make_labels_strictly_increasing(old_segment.labels, static_cast<MeshScalar>(1) / denom);
    if (tied) {
        bias_tied_segment_labels(old_segment.labels);
    }

    const auto [min_iter, max_iter] = std::minmax_element(old_segment.labels.begin(), old_segment.labels.end());
    NewPathWork arc_work = extract_cyclic_new_arc(new_path, *min_iter, *max_iter);
    if (arc_work.ids.empty()) {
        arc_work = new_work;
    } else if (old_segment.closed && new_path.closed && !arc_work.closed
               && new_work.ids.size() > arc_work.ids.size()
               && new_work.ids.size() - arc_work.ids.size() <= 4) {
        // A complete old contour can leave a few new seam samples outside
        // the half-step label window. Restore only those samples at the local
        // seam and close the arc so the complete loop can be stitched.
        NewPathWork extended = restore_seam_samples(new_work, arc_work);
        if (!extended.ids.empty()) {
            arc_work = std::move(extended);
        }
    } else if (!old_segment.closed && arc_work.closed) {
        // A partial old segment may span the whole numeric period without
        // being the complete old contour. Keep the selected new arc local,
        // but never expose its cyclic closing edge to that partial segment.
        arc_work = make_new_path_work_from_arc(
            std::move(arc_work.ids), std::move(arc_work.labels), false
        );
    }
    return PreparedStitchPaths{
        std::move(old_segment), std::move(arc_work), collapsed
    };
}

void add_connecting_edge(
    std::vector<std::array<int, 2>>& edges,
    int old_index,
    int new_index
)
{
    const std::array<int, 2> edge{old_index, new_index};
    if (std::find(edges.begin(), edges.end(), edge) == edges.end()) {
        edges.push_back(edge);
    }
}

void add_closed_seam_connection(
    std::vector<std::array<int, 2>>& connecting,
    const PreparedStitchPaths& prepared,
    const StitchConnectionOracle* connection_oracle
)
{
    if (!prepared.old_segment.closed
        || prepared.old_segment.ids.size() < 3
        || prepared.new_work.ids.size() < 3
        || !prepared.new_work.closed) {
        return;
    }

    // The zipper already connects the two path endpoints linearly. Add the
    // seam diagonal so the old and new closing edges can form the final quad.
    const int old_idx = static_cast<int>(prepared.old_segment.ids.size() - 1);
    if (connection_oracle == nullptr || connection_oracle->pair_allowed_for_geometry(
            prepared.old_segment.ids[static_cast<std::size_t>(old_idx)],
            prepared.new_work.ids.front())) {
        add_connecting_edge(connecting, old_idx, 0);
    }
}

std::vector<std::array<int, 2>> make_connecting_edges(
    const PreparedStitchPaths& prepared,
    const StitchConnectionOracle* connection_oracle
)
{
    std::vector<std::array<int, 2>> connecting = greedy_connecting_edge_indices(
        prepared.old_segment.labels,
        prepared.new_work.labels
    );
    add_closed_seam_connection(connecting, prepared, connection_oracle);
    connecting.erase(
        std::remove_if(
            connecting.begin(), connecting.end(),
            [&](const std::array<int, 2>& edge) {
                if (connection_oracle != nullptr && !connection_oracle->pair_allowed_for_geometry(
                        prepared.old_segment.ids[static_cast<std::size_t>(edge[0])],
                        prepared.new_work.ids[static_cast<std::size_t>(edge[1])])) {
                    return true;
                }
                return false;
            }),
        connecting.end());
    return connecting;
}

std::vector<Triangle> triangles_from_local_stitch_graph(
    const std::vector<int>& old_ids,
    const NewPathWork& new_work,
    const std::vector<std::array<int, 2>>& stitch_edges,
    bool old_closed
)
{
    std::unordered_set<std::uint64_t> stitch_edge_keys;
    stitch_edge_keys.reserve(stitch_edges.size());
    for (const auto edge : stitch_edges) {
        stitch_edge_keys.insert(compact_stitch_vertex_key(
            old_ids[static_cast<std::size_t>(edge[0])],
            new_work.ids[static_cast<std::size_t>(edge[1])]
        ));
    }

    std::unordered_map<int, std::vector<int>> old_neighbors;
    old_neighbors.reserve(old_ids.size());
    for (std::size_t idx = 0; idx + 1 < old_ids.size(); ++idx) {
        add_neighbor(old_neighbors, old_ids[idx], old_ids[idx + 1]);
    }
    if (old_closed && old_ids.size() >= 3) {
        add_neighbor(old_neighbors, old_ids.back(), old_ids.front());
    }

    std::unordered_set<std::uint64_t> seen;
    seen.reserve(stitch_edges.size() * 2);
    std::vector<Triangle> faces;
    faces.reserve(stitch_edges.size() * 2);
    const auto add_face = [&](int a, int b, int c) {
        const auto key = face_key(Triangle{a, b, c});
        if (!seen.insert(compact_face_key(key)).second) {
            return;
        }
        faces.push_back(Triangle{key[0], key[1], key[2]});
    };
    const auto has_stitch_edge = [&](int old_id, int new_id) {
        return stitch_edge_keys.contains(compact_stitch_vertex_key(old_id, new_id));
    };

    for (const auto edge : stitch_edges) {
        const int old_id = old_ids[static_cast<std::size_t>(edge[0])];
        const int new_id = new_work.ids[static_cast<std::size_t>(edge[1])];
        if (const auto found = old_neighbors.find(old_id); found != old_neighbors.end()) {
            for (const int neighbor_old_id : found->second) {
                if (has_stitch_edge(neighbor_old_id, new_id)) {
                    add_face(old_id, new_id, neighbor_old_id);
                }
            }
        }
        if (const auto found = new_work.neighbors.find(new_id); found != new_work.neighbors.end()) {
            for (const int neighbor_new_id : found->second) {
                if (has_stitch_edge(old_id, neighbor_new_id)) {
                    add_face(old_id, new_id, neighbor_new_id);
                }
            }
        }
    }
    return faces;
}

NewPathWork make_new_path_work(const BoundaryPath& new_path)
{
    NewPathWork work;
    if (new_path.ids.size() < 2) {
        return work;
    }
    work.ids = new_path.ids;
    work.labels.resize(new_path.ids.size());
    const MeshScalar denom = static_cast<MeshScalar>(new_path.ids.size() - 1);
    for (std::size_t idx = 0; idx < work.labels.size(); ++idx) {
        work.labels[idx] = static_cast<MeshScalar>(idx) / denom;
    }
    // This is the fallback work used by linear old segments. A cyclic edge
    // is only valid after prepare_stitch_paths has established that both
    // sides cover a complete closed path and has built an explicit arc work.
    work.closed = false;
    work.edges = path_edges(new_path.ids, false);
    work.neighbors.reserve(new_path.ids.size());
    for (const Edge edge : work.edges) {
        add_neighbor(work.neighbors, edge[0], edge[1]);
    }
    return work;
}

void append_candidate_faces_for_segment(
    std::vector<Triangle>& candidates,
    const BoundaryPath& new_path,
    const NewPathWork& new_work,
    const OldLabelSegment& old_segment,
    const std::vector<Vec3f>& vertices,
    const StitchConnectionOracle* connection_oracle
)
{
    if (new_path.ids.size() < 2) {
        return;
    }
    const PreparedStitchPaths prepared = prepare_stitch_paths(
        new_path, new_work, old_segment, vertices
    );
    if (prepared.old_segment.ids.size() < 2 || prepared.new_work.ids.size() < 2) {
        return;
    }
    const std::vector<std::array<int, 2>> connecting =
        make_connecting_edges(prepared, connection_oracle);
    const std::vector<Triangle> faces = triangles_from_local_stitch_graph(
        prepared.old_segment.ids, prepared.new_work, connecting, prepared.old_segment.closed
    );
    candidates.insert(candidates.end(), faces.begin(), faces.end());
}

void append_candidate_faces_for_prepared_segment(
    std::vector<Triangle>& candidates,
    const PreparedStitchPaths& prepared,
    const StitchConnectionOracle* connection_oracle
)
{
    if (prepared.old_segment.ids.size() < 2 || prepared.new_work.ids.size() < 2) {
        return;
    }
    const std::vector<std::array<int, 2>> connecting =
        make_connecting_edges(prepared, connection_oracle);
    const std::vector<Triangle> faces = triangles_from_local_stitch_graph(
        prepared.old_segment.ids, prepared.new_work, connecting, prepared.old_segment.closed
    );
    candidates.insert(candidates.end(), faces.begin(), faces.end());
}

void add_proxy_edge(std::vector<Edge>& edges, std::unordered_set<std::uint64_t>& seen, int a, int b)
{
    if (a < 0 || b < 0 || a == b) {
        return;
    }
    if (seen.insert(edge_key(a, b)).second) {
        edges.push_back({a, b});
    }
}

void add_nearest_label_proxy_edges(
    std::vector<Edge>& edges,
    std::unordered_set<std::uint64_t>& seen,
    const OldLabelSegment& old_segment,
    const NewPathWork& new_work,
    const StitchConnectionOracle* connection_oracle
)
{
    if (old_segment.ids.empty() || new_work.ids.empty() || new_work.labels.empty()) {
        return;
    }
    // Prepared stitch paths normalize both label sequences to monotonic order.
    std::size_t hi = 0;
    for (std::size_t idx = 0; idx < old_segment.ids.size(); ++idx) {
        const MeshScalar value = old_segment.labels[idx];
        while (hi < new_work.labels.size() && new_work.labels[hi] < value) {
            ++hi;
        }
        std::size_t new_idx = 0;
        if (hi == 0) {
            new_idx = 0;
        } else if (hi >= new_work.labels.size()) {
            new_idx = new_work.labels.size() - 1;
        } else {
            const std::size_t lo = hi - 1;
            new_idx = std::abs(new_work.labels[hi] - value) < std::abs(value - new_work.labels[lo]) ? hi : lo;
        }
        const Edge edge{old_segment.ids[idx], new_work.ids[new_idx]};
        if (connection_oracle != nullptr
            && !connection_oracle->pair_allowed_for_geometry(edge[0], edge[1])) {
            continue;
        }
        add_proxy_edge(edges, seen, edge[0], edge[1]);
    }
}

std::vector<PreparedSegmentWork> build_prepared_segment_work(
    const std::vector<BoundaryPath>& new_paths,
    const std::vector<NewPathWork>& new_path_work,
    const std::vector<std::vector<OldLabelSegment>>& old_segments_by_cluster,
    const std::vector<std::vector<bool>>& suppressed_segments,
    const std::vector<Vec3f>& vertices
)
{
    std::vector<PreparedSegmentWork> work;
    std::size_t segment_count = 0;
    for (const std::vector<OldLabelSegment>& segments : old_segments_by_cluster) {
        segment_count += segments.size();
    }
    work.reserve(segment_count);
    for (std::size_t cluster = 0; cluster < old_segments_by_cluster.size(); ++cluster) {
        if (cluster >= new_paths.size() || cluster >= new_path_work.size()) {
            continue;
        }
        const std::vector<OldLabelSegment>& segments = old_segments_by_cluster[cluster];
        for (std::size_t segment = 0; segment < segments.size(); ++segment) {
            if (cluster < suppressed_segments.size()
                && segment < suppressed_segments[cluster].size()
                && suppressed_segments[cluster][segment]) {
                continue;
            }
            PreparedStitchPaths prepared = prepare_stitch_paths(
                new_paths[cluster], new_path_work[cluster], segments[segment], vertices
            );
            if (prepared.old_segment.ids.size() < 2 || prepared.new_work.ids.size() < 2) {
                continue;
            }
            work.push_back(PreparedSegmentWork{
                static_cast<int>(cluster), false, {},
                std::move(prepared)
            });
        }
    }
    return work;
}

struct SegmentEndpoint {
    int vertex{-1};
    MeshScalar label{};
};

std::array<SegmentEndpoint, 2> segment_endpoints(const PreparedSegmentWork& segment)
{
    const OldLabelSegment& old = segment.paths.old_segment;
    return {
        SegmentEndpoint{old.ids.front(), old.labels.front()},
        SegmentEndpoint{old.ids.back(), old.labels.back()},
    };
}

std::optional<std::array<SegmentEndpoint, 2>> find_bridge_endpoints(
    const BoundaryPath& old_path,
    const PreparedSegmentWork& first,
    const PreparedSegmentWork& second
)
{
    const auto first_endpoints = segment_endpoints(first);
    const auto second_endpoints = segment_endpoints(second);
    for (const SegmentEndpoint first_endpoint : first_endpoints) {
        for (const SegmentEndpoint second_endpoint : second_endpoints) {
            if (!path_contains_edge(old_path, first_endpoint.vertex, second_endpoint.vertex)) {
                continue;
            }
            if (std::abs(first_endpoint.label - second_endpoint.label) <= kTiedLabelEpsilon) {
                continue;
            }
            if (first_endpoint.label < second_endpoint.label) {
                return std::array<SegmentEndpoint, 2>{first_endpoint, second_endpoint};
            }
            return std::array<SegmentEndpoint, 2>{second_endpoint, first_endpoint};
        }
    }
    return std::nullopt;
}

std::optional<PreparedSegmentWork> make_local_bridge(
    const std::vector<BoundaryPath>& new_paths,
    const std::vector<NewPathWork>& new_path_work,
    const std::vector<BoundaryPath>& old_paths,
    const std::vector<Vec3f>& vertices,
    const PreparedSegmentWork& first,
    const PreparedSegmentWork& second
)
{
    if (first.cluster != second.cluster
        || first.paths.old_segment.source_path < 0
        || first.paths.old_segment.source_path != second.paths.old_segment.source_path
        || first.paths.old_segment.component == second.paths.old_segment.component
        || static_cast<std::size_t>(first.cluster) >= new_paths.size()
        || static_cast<std::size_t>(first.cluster) >= new_path_work.size()
        || static_cast<std::size_t>(first.paths.old_segment.source_path) >= old_paths.size()) {
        return std::nullopt;
    }
    const BoundaryPath& old_path = old_paths[
        static_cast<std::size_t>(first.paths.old_segment.source_path)
    ];
    if (old_path.closed) {
        return std::nullopt;
    }
    const auto endpoints = find_bridge_endpoints(old_path, first, second);
    if (!endpoints.has_value()) {
        return std::nullopt;
    }

    OldLabelSegment old_bridge{
        {(*endpoints)[0].vertex, (*endpoints)[1].vertex},
        {(*endpoints)[0].label, (*endpoints)[1].label},
        false,
        first.paths.old_segment.component,
        first.paths.old_segment.source_path,
    };
    PreparedStitchPaths prepared = prepare_stitch_paths(
        new_paths[static_cast<std::size_t>(first.cluster)],
        new_path_work[static_cast<std::size_t>(first.cluster)],
        std::move(old_bridge),
        vertices
    );
    constexpr std::size_t kMaxBridgeInteriorSamples = 2;
    if (prepared.new_work.ids.size() < 3
        || prepared.new_work.ids.size() - 2 > kMaxBridgeInteriorSamples) {
        return std::nullopt;
    }
    std::vector<int> replaced_new_vertices(
        prepared.new_work.ids.begin() + 1, prepared.new_work.ids.end() - 1
    );
    return PreparedSegmentWork{
        first.cluster,
        true,
        std::move(replaced_new_vertices),
        std::move(prepared),
    };
}

std::vector<PreparedSegmentWork> build_local_bridges(
    const std::vector<BoundaryPath>& new_paths,
    const std::vector<NewPathWork>& new_path_work,
    const std::vector<BoundaryPath>& old_paths,
    const std::vector<Vec3f>& vertices,
    const std::vector<PreparedSegmentWork>& prepared_segments
)
{
    std::vector<PreparedSegmentWork> bridges;
    for (std::size_t first = 0; first < prepared_segments.size(); ++first) {
        for (std::size_t second = first + 1; second < prepared_segments.size(); ++second) {
            const auto bridge = make_local_bridge(
                new_paths,
                new_path_work,
                old_paths,
                vertices,
                prepared_segments[first],
                prepared_segments[second]
            );
            if (bridge.has_value()) {
                bridges.push_back(std::move(*bridge));
            }
        }
    }
    return bridges;
}

bool has_junction_support(
    const std::vector<JunctionSegmentSupport>& supports,
    int segment,
    int junction
)
{
    return std::any_of(
        supports.begin(), supports.end(),
        [&](const JunctionSegmentSupport& support) {
            return support.segment == segment && support.junction == junction;
        }
    );
}

bool has_distinct_branch(
    const JunctionSegmentSupport& first,
    const JunctionSegmentSupport& second
)
{
    return std::any_of(
        first.branches.begin(), first.branches.end(),
        [&](const BoundaryHalfEdge& first_branch) {
            return std::find_if(
                second.branches.begin(), second.branches.end(),
                [&](const BoundaryHalfEdge& second_branch) {
                    return same_half_edge(first_branch, second_branch);
                }
            ) == second.branches.end();
        }
    );
}

std::vector<BoundaryHalfEdge> branches_for_segment(
    const PreparedSegmentWork& segment,
    const OldJunction& junction
)
{
    std::vector<BoundaryHalfEdge> branches;
    for (const BoundaryHalfEdge& branch : junction.branches) {
        if (branch.path != segment.paths.old_segment.source_path) {
            continue;
        }
        if (std::find(
                segment.paths.old_segment.ids.begin(),
                segment.paths.old_segment.ids.end(),
                branch.neighbor
            ) != segment.paths.old_segment.ids.end()) {
            branches.push_back(branch);
        }
    }
    return branches;
}

JunctionSupportMap build_junction_supports(
    const std::vector<PreparedSegmentWork>& prepared_segments,
    const std::vector<OldJunction>& junctions,
    std::size_t cluster_count
)
{
    JunctionSupportMap supports(cluster_count);
    for (std::size_t segment_index = 0;
         segment_index < prepared_segments.size(); ++segment_index) {
        const PreparedSegmentWork& segment = prepared_segments[segment_index];
        if (segment.local_bridge || segment.cluster < 0
            || static_cast<std::size_t>(segment.cluster) >= supports.size()) {
            continue;
        }
        for (const OldJunction& junction : junctions) {
            const std::vector<BoundaryHalfEdge> branches = branches_for_segment(segment, junction);
            if (branches.empty()) {
                continue;
            }
            for (const int new_vertex : segment.paths.new_work.ids) {
                supports[static_cast<std::size_t>(segment.cluster)][new_vertex].push_back(
                    JunctionSegmentSupport{
                        static_cast<int>(segment_index), junction.vertex, branches
                    }
                );
            }
        }
    }
    return supports;
}

using SourceCostMap = std::unordered_map<int, MeshScalar>;

MeshScalar segment_vertex_cost(
    const PreparedStitchPaths& prepared,
    std::size_t new_position,
    const std::vector<Vec3f>& vertices
)
{
    if (new_position >= prepared.new_work.ids.size()
        || prepared.old_segment.ids.empty()) {
        return std::numeric_limits<MeshScalar>::infinity();
    }
    const MeshScalar target_label = prepared.new_work.labels[new_position];
    MeshScalar best_label_delta = std::numeric_limits<MeshScalar>::infinity();
    MeshScalar best_distance = std::numeric_limits<MeshScalar>::infinity();
    for (std::size_t index = 0; index < prepared.old_segment.ids.size(); ++index) {
        const MeshScalar label_delta = std::abs(prepared.old_segment.labels[index] - target_label);
        const int old_id = prepared.old_segment.ids[index];
        const int new_id = prepared.new_work.ids[new_position];
        if (old_id < 0 || new_id < 0
            || static_cast<std::size_t>(old_id) >= vertices.size()
            || static_cast<std::size_t>(new_id) >= vertices.size()) {
            continue;
        }
        const Vec3f old_point = vertices[static_cast<std::size_t>(old_id)];
        const Vec3f new_point = vertices[static_cast<std::size_t>(new_id)];
        const MeshScalar dx = old_point.x - new_point.x;
        const MeshScalar dy = old_point.y - new_point.y;
        const MeshScalar dz = old_point.z - new_point.z;
        const MeshScalar distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (label_delta < best_label_delta - kTiedLabelEpsilon
            || (std::abs(label_delta - best_label_delta) <= kTiedLabelEpsilon
                && distance < best_distance)) {
            best_label_delta = label_delta;
            best_distance = distance;
        }
    }
    return best_distance;
}

std::vector<int> ownership_states(const SourceCostMap& costs)
{
    if (costs.empty()) {
        return {-1};
    }
    std::vector<int> states;
    states.reserve(costs.size());
    for (const auto& entry : costs) {
        states.push_back(entry.first);
    }
    std::sort(states.begin(), states.end());
    return states;
}

MeshScalar ownership_state_cost(const SourceCostMap& costs, int source_path)
{
    if (source_path < 0) {
        return static_cast<MeshScalar>(0);
    }
    const auto found = costs.find(source_path);
    return found == costs.end() ? std::numeric_limits<MeshScalar>::infinity() : found->second;
}

// Keep the cyclic dynamic program together: the transition cost couples every
// boundary sample to its predecessor and the closed-path seam.
std::vector<int> choose_owner_sequence(
    const std::vector<SourceCostMap>& costs,
    bool closed
)
{
    if (costs.empty()) {
        return {};
    }
    std::vector<std::vector<int>> states;
    states.reserve(costs.size());
    for (const SourceCostMap& position_costs : costs) {
        states.push_back(ownership_states(position_costs));
    }

    constexpr MeshScalar kOwnerSwitchPenalty = static_cast<MeshScalar>(0.02);
    const std::size_t position_count = states.size();
    MeshScalar best_total = std::numeric_limits<MeshScalar>::infinity();
    std::vector<int> best_sequence;
    std::vector<std::vector<MeshScalar>> dynamic_cost(position_count);
    std::vector<std::vector<int>> previous_state(position_count);
    for (std::size_t position = 0; position < position_count; ++position) {
        dynamic_cost[position].resize(states[position].size());
        previous_state[position].resize(states[position].size());
    }
    for (std::size_t start_state = 0; start_state < states.front().size(); ++start_state) {
        for (std::size_t position = 0; position < position_count; ++position) {
            std::fill(
                dynamic_cost[position].begin(), dynamic_cost[position].end(),
                std::numeric_limits<MeshScalar>::infinity()
            );
            std::fill(previous_state[position].begin(), previous_state[position].end(), -1);
        }
        dynamic_cost[0][start_state] = ownership_state_cost(
            costs[0], states[0][start_state]
        );
        for (std::size_t position = 1; position < position_count; ++position) {
            for (std::size_t current = 0; current < states[position].size(); ++current) {
                const int current_owner = states[position][current];
                const MeshScalar current_cost = ownership_state_cost(
                    costs[position], current_owner
                );
                for (std::size_t previous = 0; previous < states[position - 1].size(); ++previous) {
                    const int previous_owner = states[position - 1][previous];
                    const MeshScalar transition = previous_owner == current_owner
                        ? static_cast<MeshScalar>(0) : kOwnerSwitchPenalty;
                    const MeshScalar total = dynamic_cost[position - 1][previous]
                        + transition + current_cost;
                    if (total < dynamic_cost[position][current]) {
                        dynamic_cost[position][current] = total;
                        previous_state[position][current] = static_cast<int>(previous);
                    }
                }
            }
        }

        int final_state = -1;
        MeshScalar total = std::numeric_limits<MeshScalar>::infinity();
        for (std::size_t state = 0; state < states.back().size(); ++state) {
            const int end_owner = states.back()[state];
            const int start_owner = states.front()[start_state];
            const MeshScalar wrap_penalty = closed && end_owner != start_owner
                ? kOwnerSwitchPenalty : static_cast<MeshScalar>(0);
            const MeshScalar candidate = dynamic_cost.back()[state] + wrap_penalty;
            if (candidate < total) {
                total = candidate;
                final_state = static_cast<int>(state);
            }
        }
        if (final_state < 0 || total >= best_total) {
            continue;
        }
        std::vector<int> sequence(position_count, -1);
        int state = final_state;
        for (std::size_t position = position_count; position-- > 0;) {
            sequence[position] = states[position][static_cast<std::size_t>(state)];
            state = previous_state[position][static_cast<std::size_t>(state)];
            if (position > 0 && state < 0) {
                sequence.clear();
                break;
            }
        }
        if (!sequence.empty()) {
            best_total = total;
            best_sequence = std::move(sequence);
        }
    }
    return best_sequence;
}

void begin_ownership_workspace(
    CandidateOwnershipWorkspace& workspace,
    const std::vector<BoundaryPath>& new_paths,
    const std::vector<Vec3f>& vertices
)
{
    std::size_t required_size = vertices.size();
    for (const BoundaryPath& path : new_paths) {
        for (const int vertex : path.ids) {
            if (vertex >= 0) {
                required_size = std::max(required_size, static_cast<std::size_t>(vertex) + 1);
            }
        }
    }
    if (workspace.owners.size() < required_size) {
        workspace.owners.resize(required_size);
        workspace.generations.resize(required_size, 0);
    }
    ++workspace.generation;
    if (workspace.generation == 0) {
        std::fill(workspace.generations.begin(), workspace.generations.end(), 0);
        workspace.generation = 1;
    }
}

void set_ownership(CandidateOwnershipWorkspace& workspace, int vertex, int owner)
{
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= workspace.owners.size()) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(vertex);
    workspace.owners[index] = owner;
    workspace.generations[index] = workspace.generation;
}

bool has_ownership(const CandidateOwnershipWorkspace& workspace, int vertex)
{
    return vertex >= 0
        && static_cast<std::size_t>(vertex) < workspace.generations.size()
        && workspace.generations[static_cast<std::size_t>(vertex)] == workspace.generation;
}

void build_new_vertex_ownership(
    const std::vector<BoundaryPath>& new_paths,
    const std::vector<PreparedSegmentWork>& prepared_segments,
    const std::vector<Vec3f>& vertices,
    CandidateOwnershipWorkspace& ownership,
    PipelineTiming* timing
)
{
    Stopwatch stopwatch;
    begin_ownership_workspace(ownership, new_paths, vertices);
    if (timing != nullptr) {
        timing->add_stage(
            "stitch.core.prepare_work.ownership.begin_workspace",
            stopwatch.elapsed_milliseconds()
        );
    }

    stopwatch = Stopwatch{};
    std::vector<std::vector<SourceCostMap>> costs(new_paths.size());
    std::vector<std::unordered_map<int, std::size_t>> positions(new_paths.size());
    for (std::size_t cluster = 0; cluster < new_paths.size(); ++cluster) {
        costs[cluster].resize(new_paths[cluster].ids.size());
        positions[cluster].reserve(new_paths[cluster].ids.size());
        for (std::size_t position = 0; position < new_paths[cluster].ids.size(); ++position) {
            set_ownership(ownership, new_paths[cluster].ids[position], -1);
            positions[cluster][new_paths[cluster].ids[position]] = position;
        }
    }
    if (timing != nullptr) {
        timing->add_stage(
            "stitch.core.prepare_work.ownership.initialize_paths",
            stopwatch.elapsed_milliseconds()
        );
    }

    stopwatch = Stopwatch{};
    for (const PreparedSegmentWork& prepared : prepared_segments) {
        const int source_path = prepared.paths.old_segment.source_path;
        const int cluster = prepared.cluster;
        if (source_path < 0 || cluster < 0
            || static_cast<std::size_t>(cluster) >= positions.size()) {
            continue;
        }
        for (std::size_t new_position = 0;
             new_position < prepared.paths.new_work.ids.size(); ++new_position) {
            const auto found = positions[static_cast<std::size_t>(cluster)].find(
                prepared.paths.new_work.ids[new_position]
            );
            if (found == positions[static_cast<std::size_t>(cluster)].end()) {
                continue;
            }
            const MeshScalar cost = segment_vertex_cost(prepared.paths, new_position, vertices);
            auto& source_costs = costs[static_cast<std::size_t>(cluster)][found->second];
            const auto existing = source_costs.find(source_path);
            if (existing == source_costs.end() || cost < existing->second) {
                source_costs[source_path] = cost;
            }
        }
    }
    if (timing != nullptr) {
        timing->add_stage(
            "stitch.core.prepare_work.ownership.populate_costs",
            stopwatch.elapsed_milliseconds()
        );
    }

    stopwatch = Stopwatch{};
    std::vector<std::vector<int>> owner_sequences(new_paths.size());
#pragma omp parallel for if (new_paths.size() > 8) schedule(guided)
    for (int cluster_index = 0; cluster_index < static_cast<int>(new_paths.size());
         ++cluster_index) {
        const std::size_t cluster = static_cast<std::size_t>(cluster_index);
        owner_sequences[cluster] = choose_owner_sequence(
            costs[cluster], new_paths[cluster].closed
        );
    }
    if (timing != nullptr) {
        timing->add_stage(
            "stitch.core.prepare_work.ownership.solve_sequences",
            stopwatch.elapsed_milliseconds()
        );
    }

    stopwatch = Stopwatch{};
    for (std::size_t cluster = 0; cluster < new_paths.size(); ++cluster) {
        const std::vector<int>& owners = owner_sequences[cluster];
        for (std::size_t position = 0; position < owners.size(); ++position) {
            set_ownership(ownership, new_paths[cluster].ids[position], owners[position]);
        }
    }
    if (timing != nullptr) {
        timing->add_stage(
            "stitch.core.prepare_work.ownership.publish_sequences",
            stopwatch.elapsed_milliseconds()
        );
    }
}

std::vector<Edge> build_polyline_proxy_edges(
    const std::vector<PreparedSegmentWork>& prepared_segments,
    const StitchConnectionOracle* connection_oracle
)
{
    std::vector<Edge> edges;
    std::unordered_set<std::uint64_t> seen;
    for (const PreparedSegmentWork& segment : prepared_segments) {
        const PreparedStitchPaths& prepared = segment.paths;
        edges.reserve(edges.size() + prepared.old_segment.ids.size() + prepared.new_work.edges.size() + 4);
        seen.reserve(edges.capacity() * 2);
        for (std::size_t idx = 0; idx + 1 < prepared.old_segment.ids.size(); ++idx) {
            add_proxy_edge(edges, seen, prepared.old_segment.ids[idx], prepared.old_segment.ids[idx + 1]);
        }
        if (prepared.old_segment.closed
            && prepared.old_segment.ids.size() >= 3
            && prepared.new_work.closed) {
            const Edge old_seam{
                prepared.old_segment.ids.back(), prepared.old_segment.ids.front()};
            add_proxy_edge(edges, seen, old_seam[0], old_seam[1]);
            const Edge new_seam{
                prepared.old_segment.ids.back(), prepared.new_work.ids.front()};
            if ((connection_oracle == nullptr || connection_oracle->pair_allowed_for_geometry(
                    new_seam[0], new_seam[1]))
            ) {
                add_proxy_edge(edges, seen, new_seam[0], new_seam[1]);
            }
        }
        for (const Edge edge : prepared.new_work.edges) {
            add_proxy_edge(edges, seen, edge[0], edge[1]);
        }
        add_nearest_label_proxy_edges(
            edges, seen, prepared.old_segment, prepared.new_work,
            connection_oracle);
    }
    return edges;
}

}  // namespace

namespace {

void add_timing(PipelineTiming* timing, const char* name, const Stopwatch& stopwatch)
{
    if (timing != nullptr) {
        timing->add_stage(name, stopwatch.elapsed_milliseconds());
    }
}

class CandidateGenerator {
public:
    CandidateGenerator(
        const CandidateGenerationRequest& request,
        StitchCoreDebug* debug,
        PipelineTiming* timing
    )
        : request_(request), debug_(debug), timing_(timing)
    {
        ownership_workspace_ = request_.ownership_workspace != nullptr
            ? request_.ownership_workspace
            : &local_ownership_workspace_;
    }

    std::vector<Triangle> run()
    {
        build_new_paths();
        Stopwatch stopwatch;
        prepare_collapsed_seam_repairs();
        add_timing(timing_, "stitch.core.prepare_collapsed_seam_repairs", stopwatch);
        stopwatch = Stopwatch{};
        prepare_work();
        add_timing(timing_, "stitch.core.prepare_work", stopwatch);
        generate_chunks();
        if (!request_.proxy_edges_callback) {
            Stopwatch closure_stopwatch;
            stopwatch = Stopwatch{};
            prepare_junction_closures();
            add_timing(timing_, "stitch.core.prepare_junction_closures", stopwatch);
            stopwatch = Stopwatch{};
            prepare_collapsed_interval_closures();
            add_timing(timing_, "stitch.core.prepare_collapsed_interval_closures", stopwatch);
            add_timing(timing_, "stitch.core.prepare_closures", closure_stopwatch);
        }
        return merge_chunks();
    }

private:
    void build_new_paths()
    {
        Stopwatch stopwatch;
        new_path_work_.reserve(request_.new_paths.size());
        for (const BoundaryPath& new_path : request_.new_paths) {
            new_path_work_.push_back(make_new_path_work(new_path));
        }
        add_timing(timing_, "stitch.core.new_path_work", stopwatch);
    }

    void prepare_work()
    {
        Stopwatch stopwatch;
        prepared_segments_ = build_prepared_segment_work(
            request_.new_paths,
            new_path_work_,
            request_.old_segments_by_cluster,
            suppressed_segments_,
            request_.vertices
        );
        add_timing(timing_, "stitch.core.prepare_work.segments", stopwatch);
        if (!request_.proxy_edges_callback && request_.old_paths != nullptr) {
            stopwatch = Stopwatch{};
            std::vector<PreparedSegmentWork> bridges = build_local_bridges(
                request_.new_paths,
                new_path_work_,
                *request_.old_paths,
                request_.vertices,
                prepared_segments_
            );
            for (const PreparedSegmentWork& bridge : bridges) {
                replaced_new_vertices_.insert(
                    bridge.replaced_new_vertices.begin(),
                    bridge.replaced_new_vertices.end()
                );
            }
            prepared_segments_.insert(
                prepared_segments_.end(),
                std::make_move_iterator(bridges.begin()),
                std::make_move_iterator(bridges.end())
            );
            add_timing(timing_, "stitch.core.prepare_work.bridges", stopwatch);
        }
        stopwatch = Stopwatch{};
        build_new_vertex_ownership(
            request_.new_paths, prepared_segments_, request_.vertices, *ownership_workspace_, timing_
        );
        add_timing(timing_, "stitch.core.prepare_work.ownership", stopwatch);
        if (request_.proxy_edges_callback) {
            prepare_proxy_work();
        } else {
            prepare_candidate_tasks();
        }
    }

    void prepare_junction_closures()
    {
        if (request_.old_paths == nullptr || prepared_segments_.empty()) {
            return;
        }
        const std::vector<OldJunction> junctions = build_old_junction_incidence_graph(
            *request_.old_paths
        );
        if (junctions.empty()) {
            return;
        }
        const JunctionSupportMap supports = build_junction_supports(
            prepared_segments_, junctions, request_.new_paths.size()
        );
        for (std::size_t cluster = 0; cluster < request_.new_paths.size(); ++cluster) {
            append_junction_cells_for_path(cluster, supports);
        }
        remove_competing_candidate_faces();
    }

    void append_junction_cells_for_path(
        std::size_t cluster,
        const JunctionSupportMap& supports
    )
    {
        const BoundaryPath& new_path = request_.new_paths[cluster];
        if (new_path.ids.size() < 3 || !new_path.closed) {
            return;
        }
        for (std::size_t position = 0; position < new_path.ids.size(); ++position) {
            const std::size_t previous = position == 0
                ? new_path.ids.size() - 1 : position - 1;
            const std::size_t next = (position + 1) % new_path.ids.size();
            const auto previous_supports = supports[cluster].find(new_path.ids[previous]);
            const auto current_supports = supports[cluster].find(new_path.ids[position]);
            const auto next_supports = supports[cluster].find(new_path.ids[next]);
            if (previous_supports == supports[cluster].end()
                || current_supports == supports[cluster].end()
                || next_supports == supports[cluster].end()) {
                continue;
            }
            append_junction_cell_candidates(
                new_path.ids[previous],
                new_path.ids[position],
                new_path.ids[next],
                previous_supports->second,
                current_supports->second,
                next_supports->second
            );
        }
    }

    void append_junction_cell_candidates(
        int previous_new,
        int current_new,
        int next_new,
        const std::vector<JunctionSegmentSupport>& previous_supports,
        const std::vector<JunctionSegmentSupport>& current_supports,
        const std::vector<JunctionSegmentSupport>& next_supports
    )
    {
        for (const JunctionSegmentSupport& left : previous_supports) {
            for (const JunctionSegmentSupport& right : next_supports) {
                if (left.segment == right.segment || left.junction != right.junction
                    || !has_distinct_branch(left, right)
                    || !has_junction_support(
                        current_supports, left.segment, left.junction
                    )
                    || !has_junction_support(
                        current_supports, right.segment, right.junction
                    )
                    || has_junction_support(
                        next_supports, left.segment, left.junction
                    )
                    || has_junction_support(
                        previous_supports, right.segment, right.junction
                    )) {
                    continue;
                }
                append_junction_cell(
                    left,
                    right,
                    previous_new,
                    current_new,
                    next_new
                );
            }
        }
    }

    void append_junction_cell(
        const JunctionSegmentSupport& left,
        const JunctionSegmentSupport& right,
        int previous_new,
        int current_new,
        int next_new
    )
    {
        if (!stitch_pair_allowed(left.junction, previous_new)
            || !stitch_pair_allowed(left.junction, current_new)
            || !stitch_pair_allowed(right.junction, next_new)) {
            return;
        }
        append_junction_face(Triangle{left.junction, previous_new, current_new});
        append_junction_face(Triangle{right.junction, current_new, next_new});
        claimed_boundary_edges_.insert(edge_key(previous_new, current_new));
        claimed_boundary_edges_.insert(edge_key(current_new, next_new));
        for (const BoundaryHalfEdge& branch : left.branches) {
            append_junction_side_face(branch, left.junction, previous_new);
        }
        for (const BoundaryHalfEdge& branch : right.branches) {
            append_junction_side_face(branch, right.junction, next_new);
        }
    }

    void append_junction_face(Triangle face)
    {
        const auto key = face_key(face);
        if (junction_face_keys_.insert(compact_face_key(key)).second) {
            junction_faces_.push_back(Triangle{key[0], key[1], key[2]});
        }
    }

    void append_junction_side_face(
        const BoundaryHalfEdge& branch,
        int junction,
        int new_vertex
    )
    {
        if (!stitch_pair_allowed(branch.neighbor, new_vertex)
            || !stitch_pair_allowed(junction, new_vertex)) {
            return;
        }
        append_junction_face(Triangle{branch.neighbor, junction, new_vertex});
        claimed_boundary_edges_.insert(edge_key(branch.junction, branch.neighbor));
    }

    void remove_competing_candidate_faces()
    {
        if (claimed_boundary_edges_.empty()) {
            return;
        }
        for (std::vector<Triangle>& chunk : candidate_chunks_) {
            chunk.erase(
                std::remove_if(
                    chunk.begin(), chunk.end(),
                    [&](const Triangle& face) {
                        const Edge edges[3]{
                            {face.v0, face.v1},
                            {face.v1, face.v2},
                            {face.v2, face.v0},
                        };
                        return std::any_of(
                            std::begin(edges), std::end(edges),
                            [&](const Edge edge) {
                                return claimed_boundary_edges_.contains(
                                    edge_key(edge[0], edge[1])
                                );
                            }
                        );
                    }
                ),
                chunk.end()
            );
        }
    }

    bool candidate_or_junction_uses_edge(int first, int second) const
    {
        return std::binary_search(
            candidate_edge_keys_.begin(), candidate_edge_keys_.end(), edge_key(first, second)
        );
    }

    int nearest_new_vertex(const NewPathWork& new_work, MeshScalar target_label) const
    {
        if (new_work.ids.empty() || new_work.ids.size() != new_work.labels.size()) {
            return -1;
        }
        int best_vertex = -1;
        MeshScalar best_distance = std::numeric_limits<MeshScalar>::infinity();
        for (std::size_t index = 0; index < new_work.ids.size(); ++index) {
            const MeshScalar distance = std::abs(new_work.labels[index] - target_label);
            if (distance < best_distance) {
                best_distance = distance;
                best_vertex = new_work.ids[index];
            }
        }
        return best_vertex;
    }

    void append_collapsed_face(Triangle face)
    {
        const auto key = face_key(face);
        if (collapsed_interval_face_keys_.insert(compact_face_key(key)).second) {
            collapsed_interval_faces_.push_back(Triangle{key[0], key[1], key[2]});
        }
    }

    void append_collapsed_interval_face(
        int first_old,
        int second_old,
        int anchor_new
    )
    {
        if (first_old < 0 || second_old < 0 || anchor_new < 0
            || first_old == second_old
            || !stitch_pair_allowed(first_old, anchor_new)
            || !stitch_pair_allowed(second_old, anchor_new)) {
            return;
        }
        append_collapsed_face(Triangle{first_old, second_old, anchor_new});
        claimed_boundary_edges_.insert(edge_key(first_old, second_old));
    }

    void prepare_collapsed_interval_closures()
    {
        Stopwatch stopwatch;
        build_candidate_edge_index();
        add_timing(timing_, "stitch.core.prepare_collapsed_interval_closures.edge_index", stopwatch);
        for (const PreparedSegmentWork& segment : prepared_segments_) {
            if (segment.local_bridge || segment.paths.collapsed_old_intervals.empty()) {
                continue;
            }
            const OldLabelSegment& old = segment.paths.old_segment;
            for (const auto interval : segment.paths.collapsed_old_intervals) {
                if (interval[1] >= old.ids.size() || interval[1] >= old.labels.size()) {
                    continue;
                }
                const int first_old = old.ids[interval[0]];
                const int second_old = old.ids[interval[1]];
                if (!candidate_or_junction_uses_edge(first_old, second_old)) {
                    const int anchor_new = nearest_new_vertex(
                        segment.paths.new_work, old.labels[interval[1]]
                    );
                    append_collapsed_interval_face(first_old, second_old, anchor_new);
                }

            }
        }
        remove_competing_candidate_faces();
    }

    void build_candidate_edge_index()
    {
        std::size_t edge_count = junction_faces_.size() * 3;
        for (const std::vector<Triangle>& chunk : candidate_chunks_) {
            edge_count += chunk.size() * 3;
        }
        candidate_edge_keys_.clear();
        candidate_edge_keys_.reserve(edge_count);
        const auto append_face_edges = [&](const Triangle face) {
            candidate_edge_keys_.push_back(edge_key(face.v0, face.v1));
            candidate_edge_keys_.push_back(edge_key(face.v1, face.v2));
            candidate_edge_keys_.push_back(edge_key(face.v2, face.v0));
        };
        for (const std::vector<Triangle>& chunk : candidate_chunks_) {
            for (const Triangle face : chunk) {
                append_face_edges(face);
            }
        }
        for (const Triangle face : junction_faces_) {
            append_face_edges(face);
        }
        std::sort(candidate_edge_keys_.begin(), candidate_edge_keys_.end());
        candidate_edge_keys_.erase(
            std::unique(candidate_edge_keys_.begin(), candidate_edge_keys_.end()),
            candidate_edge_keys_.end()
        );
    }

    bool label_for_vertex(
        const OldLabelSegment& segment,
        int vertex,
        MeshScalar& label
    ) const
    {
        for (std::size_t index = 0; index < segment.ids.size(); ++index) {
            if (segment.ids[index] == vertex) {
                label = segment.labels[index];
                return true;
            }
        }
        return false;
    }

    bool stitch_pair_allowed(int old_vertex, int new_vertex) const
    {
        if (request_.connection_oracle != nullptr
            && !request_.connection_oracle->pair_allowed_for_geometry(old_vertex, new_vertex)) {
            return false;
        }
        return has_open_side_match(
            old_vertex,
            new_vertex,
            request_.vertices,
            request_.old_incidence,
            request_.new_incidence
        );
    }

    bool find_collapsed_seam_face(
        int cluster,
        int segment_index,
        Triangle& face
    ) const
    {
        if (request_.old_paths == nullptr
            || cluster < 0
            || static_cast<std::size_t>(cluster) >= request_.new_paths.size()) {
            return false;
        }
        const OldLabelSegment& collapsed = request_.old_segments_by_cluster[
            static_cast<std::size_t>(cluster)][static_cast<std::size_t>(segment_index)
        ];
        if (!has_tied_labels(collapsed)
            || collapsed.source_path < 0
            || static_cast<std::size_t>(collapsed.source_path) >= request_.old_paths->size()) {
            return false;
        }
        const BoundaryPath& old_path = (*request_.old_paths)[
            static_cast<std::size_t>(collapsed.source_path)
        ];
        if (!old_path.closed) {
            return false;
        }
        const auto& segments = request_.old_segments_by_cluster[static_cast<std::size_t>(cluster)];
        for (std::size_t other_index = 0; other_index < segments.size(); ++other_index) {
            if (other_index == static_cast<std::size_t>(segment_index)) {
                continue;
            }
            const OldLabelSegment& other = segments[other_index];
            if (other.source_path != collapsed.source_path
                || (collapsed.component >= 0 && other.component >= 0
                    && collapsed.component != other.component)) {
                continue;
            }
            for (std::size_t collapsed_index = 0;
                 collapsed_index < collapsed.ids.size(); ++collapsed_index) {
                MeshScalar collapsed_label = static_cast<MeshScalar>(0);
                if (!label_for_vertex(
                        collapsed,
                        collapsed.ids[collapsed_index],
                        collapsed_label
                    )) {
                    continue;
                }
                for (std::size_t other_pos = 0; other_pos < other.ids.size(); ++other_pos) {
                    MeshScalar other_label = static_cast<MeshScalar>(0);
                    if (!label_for_vertex(other, other.ids[other_pos], other_label)
                        || std::abs(collapsed_label - other_label) > kTiedLabelEpsilon
                        || !path_contains_edge(
                            old_path,
                            collapsed.ids[collapsed_index],
                            other.ids[other_pos]
                        )) {
                        continue;
                    }
                    const int anchor = nearest_path_vertex(
                        request_.new_paths[static_cast<std::size_t>(cluster)],
                        collapsed_label
                    );
                    if (anchor < 0
                        || !stitch_pair_allowed(collapsed.ids[collapsed_index], anchor)
                        || !stitch_pair_allowed(other.ids[other_pos], anchor)) {
                        continue;
                    }
                    face = Triangle{
                        other.ids[other_pos], collapsed.ids[collapsed_index], anchor
                    };
                    return true;
                }
            }
        }
        return false;
    }

    void prepare_collapsed_seam_repairs()
    {
        if (request_.old_paths == nullptr || request_.proxy_edges_callback) {
            return;
        }
        suppressed_segments_.resize(request_.old_segments_by_cluster.size());
        for (std::size_t cluster = 0; cluster < request_.old_segments_by_cluster.size(); ++cluster) {
            suppressed_segments_[cluster].assign(
                request_.old_segments_by_cluster[cluster].size(), false
            );
            for (std::size_t segment = 0;
                 segment < request_.old_segments_by_cluster[cluster].size(); ++segment) {
                Triangle face{};
                if (!find_collapsed_seam_face(
                        static_cast<int>(cluster), static_cast<int>(segment), face
                    )) {
                    continue;
                }
                suppressed_segments_[cluster][segment] = true;
                seam_repair_faces_.push_back(face);
            }
        }
    }

    void prepare_proxy_work()
    {
        Stopwatch proxy_stopwatch;
        std::vector<Edge> proxy_edges =
            build_polyline_proxy_edges(
                prepared_segments_, request_.connection_oracle
            );
        request_.proxy_edges_callback(proxy_edges);
        add_timing(timing_, "stitch.core.polyline_proxy_edges", proxy_stopwatch);
    }

    void prepare_candidate_tasks()
    {
        Stopwatch stopwatch;
        for (std::size_t cluster = 0; cluster < request_.old_segments_by_cluster.size(); ++cluster) {
            const std::vector<OldLabelSegment>& segments =
                request_.old_segments_by_cluster[cluster];
            for (std::size_t segment = 0; segment < segments.size(); ++segment) {
                if (cluster < suppressed_segments_.size()
                    && segment < suppressed_segments_[cluster].size()
                    && suppressed_segments_[cluster][segment]) {
                    continue;
                }
                candidate_tasks_.push_back(
                    CandidateTask{static_cast<int>(cluster), static_cast<int>(segment)}
                );
            }
        }
        add_timing(timing_, "stitch.core.prepare_candidate_tasks", stopwatch);
    }

    void generate_chunks()
    {
        Stopwatch stopwatch;
        const std::size_t count =
            prepared_segments_.empty() ? candidate_tasks_.size() : prepared_segments_.size();
        candidate_chunks_.resize(count);
        const int task_count = static_cast<int>(count);
#if defined(RI_MESHING_HAS_OPENMP)
        #pragma omp parallel for schedule(guided)
#endif
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
            generate_chunk(static_cast<std::size_t>(task_idx));
        }
        add_timing(timing_, "stitch.core.generate_candidates", stopwatch);
    }

    void generate_chunk(std::size_t task_idx)
    {
        if (!prepared_segments_.empty()) {
            append_candidate_faces_for_prepared_segment(
                candidate_chunks_[task_idx], prepared_segments_[task_idx].paths,
                request_.connection_oracle
            );
            filter_candidate_chunk(
                candidate_chunks_[task_idx],
                prepared_segments_[task_idx].paths.old_segment.source_path,
                prepared_segments_[task_idx].local_bridge
            );
            return;
        }
        const CandidateTask task = candidate_tasks_[task_idx];
        append_candidate_faces_for_segment(
            candidate_chunks_[task_idx],
            request_.new_paths[static_cast<std::size_t>(task.cluster)],
            new_path_work_[static_cast<std::size_t>(task.cluster)],
            request_.old_segments_by_cluster[static_cast<std::size_t>(task.cluster)]
                [static_cast<std::size_t>(task.segment)],
            request_.vertices, request_.connection_oracle
        );
    }

    void filter_candidate_chunk(
        std::vector<Triangle>& candidates,
        int source_path,
        bool local_bridge
    ) const
    {
        if (source_path < 0 || ownership_workspace_->generation == 0) {
            return;
        }
        candidates.erase(
            std::remove_if(
                candidates.begin(),
                candidates.end(),
                [&](const Triangle& face) {
                    for (const int vertex : {face.v0, face.v1, face.v2}) {
                        if (!local_bridge && replaced_new_vertices_.contains(vertex)) {
                            return true;
                        }
                        if (!has_ownership(*ownership_workspace_, vertex)) {
                            continue;
                        }
                        if (ownership_workspace_->owners[static_cast<std::size_t>(vertex)]
                            != source_path) {
                            return true;
                        }
                    }
                    return false;
                }
            ),
            candidates.end()
        );
    }

    std::vector<Triangle> merge_chunks()
    {
        Stopwatch stopwatch;
        std::size_t candidate_count = 0;
        for (const std::vector<Triangle>& chunk : candidate_chunks_) {
            candidate_count += chunk.size();
        }
        std::vector<Triangle> candidates;
        candidates.reserve(candidate_count + seam_repair_faces_.size());
        for (std::vector<Triangle>& chunk : candidate_chunks_) {
            candidates.insert(candidates.end(), chunk.begin(), chunk.end());
        }
        candidates.insert(candidates.end(), junction_faces_.begin(), junction_faces_.end());
        candidates.insert(
            candidates.end(), collapsed_interval_faces_.begin(), collapsed_interval_faces_.end()
        );
        candidates.insert(candidates.end(), seam_repair_faces_.begin(), seam_repair_faces_.end());
        if (debug_ != nullptr) {
            debug_->candidate_faces_before_filter = candidates;
        }
        add_timing(timing_, "stitch.core.merge_candidates", stopwatch);
        return candidates;
    }

    const CandidateGenerationRequest& request_;
    StitchCoreDebug* debug_{};
    PipelineTiming* timing_{};
    std::vector<NewPathWork> new_path_work_;
    std::vector<PreparedSegmentWork> prepared_segments_;
    CandidateOwnershipWorkspace* ownership_workspace_{};
    CandidateOwnershipWorkspace local_ownership_workspace_;
    std::vector<CandidateTask> candidate_tasks_;
    std::vector<std::vector<Triangle>> candidate_chunks_;
    std::vector<std::vector<bool>> suppressed_segments_;
    std::vector<std::uint64_t> candidate_edge_keys_;
    std::vector<Triangle> junction_faces_;
    std::unordered_set<std::uint64_t> junction_face_keys_;
    std::vector<Triangle> collapsed_interval_faces_;
    std::unordered_set<std::uint64_t> collapsed_interval_face_keys_;
    std::unordered_set<std::uint64_t> claimed_boundary_edges_;
    std::vector<Triangle> seam_repair_faces_;
    std::unordered_set<int> replaced_new_vertices_;
};

}  // namespace

std::vector<Triangle> generate_stitch_candidates(
    const CandidateGenerationRequest& request,
    StitchCoreDebug* debug,
    PipelineTiming* timing
)
{
    return CandidateGenerator(request, debug, timing).run();
}

}  // namespace ri_meshing::stitch_detail
