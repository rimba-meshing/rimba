#include "stitch_core_components.hpp"

#include "ri_meshing/face_orientation.hpp"
#include "ri_meshing/geometry.hpp"
#include "ri_meshing/timing.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace ri_meshing::stitch_detail {
namespace {

struct KeyedTriangle {
    std::array<int, 3> key{};
    Triangle face{};
};
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

MeshScalar squared_distance(Vec3f a, Vec3f b)
{
    const MeshScalar dx = a.x - b.x;
    const MeshScalar dy = a.y - b.y;
    const MeshScalar dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

MeshScalar longest_edge_squared(Vec3f a, Vec3f b, Vec3f c)
{
    const Vec3f ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3f ac{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3f bc{c.x - b.x, c.y - b.y, c.z - b.z};
    const auto edge_sq = [](Vec3f edge) {
        return edge.x * edge.x + edge.y * edge.y + edge.z * edge.z;
    };
    return std::max({edge_sq(ab), edge_sq(ac), edge_sq(bc)});
}

bool face_within_old_new_distance(
    Triangle face,
    const std::vector<Vec3f>& vertices,
    int old_vertex_count,
    MeshScalar max_dist
)
{
    const Edge edges[3]{{face.v0, face.v1}, {face.v1, face.v2}, {face.v2, face.v0}};
    const MeshScalar max_dist_sq = max_dist * max_dist;
    for (const Edge edge : edges) {
        const bool old_new = (edge[0] < old_vertex_count && edge[1] >= old_vertex_count)
            || (edge[1] < old_vertex_count && edge[0] >= old_vertex_count);
        if (old_new && squared_distance(vertices[edge[0]], vertices[edge[1]]) > max_dist_sq) {
            return false;
        }
    }
    return true;
}

class ManifoldFaceSelector {
public:
    ManifoldFaceSelector(
        std::vector<Triangle> candidates,
        const std::vector<Edge>& old_boundary_edges,
        const std::vector<Edge>& new_boundary_edges
    )
        : candidates_(std::move(candidates)), old_boundary_edges_(old_boundary_edges),
          new_boundary_edges_(new_boundary_edges)
    {
    }

    std::vector<Triangle> run()
    {
        key_and_sort_candidates();
        return boundary_edges_are_sorted() ? select_with_sorted_edges() : select_with_edge_map();
    }

private:
    static bool edge_range_is_sorted(const std::vector<Edge>& edges)
    {
        for (std::size_t idx = 1; idx < edges.size(); ++idx) {
            if (edge_key(edges[idx - 1][0], edges[idx - 1][1]) > edge_key(edges[idx][0], edges[idx][1])) {
                return false;
            }
        }
        return true;
    }

    bool boundary_edges_are_sorted() const
    {
        return edge_range_is_sorted(old_boundary_edges_) && edge_range_is_sorted(new_boundary_edges_);
    }

    void key_and_sort_candidates()
    {
        keyed_candidates_.reserve(candidates_.size());
        for (const Triangle face : candidates_) {
            const std::array<int, 3> key = face_key(face);
            keyed_candidates_.push_back(KeyedTriangle{key, face});
        }
        std::sort(
            keyed_candidates_.begin(), keyed_candidates_.end(),
            [](const KeyedTriangle& a, const KeyedTriangle& b) { return a.key < b.key; }
        );
    }

    std::vector<std::uint64_t> merge_boundary_edge_keys() const
    {
        std::vector<std::uint64_t> keys;
        keys.reserve(old_boundary_edges_.size() + new_boundary_edges_.size());
        std::size_t old_idx = 0;
        std::size_t new_idx = 0;
        while (old_idx < old_boundary_edges_.size() || new_idx < new_boundary_edges_.size()) {
            const std::uint64_t next_key = next_boundary_key(old_idx, new_idx);
            if (keys.empty() || keys.back() != next_key) {
                keys.push_back(next_key);
            }
            advance_past_key(old_boundary_edges_, next_key, old_idx);
            advance_past_key(new_boundary_edges_, next_key, new_idx);
        }
        return keys;
    }

    std::uint64_t next_boundary_key(std::size_t old_idx, std::size_t new_idx) const
    {
        if (new_idx >= new_boundary_edges_.size()) {
            return key_at(old_boundary_edges_, old_idx);
        }
        if (old_idx >= old_boundary_edges_.size()) {
            return key_at(new_boundary_edges_, new_idx);
        }
        return std::min(key_at(old_boundary_edges_, old_idx), key_at(new_boundary_edges_, new_idx));
    }

    static std::uint64_t key_at(const std::vector<Edge>& edges, std::size_t index)
    {
        const Edge edge = edges[index];
        return edge_key(edge[0], edge[1]);
    }

    static void advance_past_key(
        const std::vector<Edge>& edges,
        std::uint64_t key,
        std::size_t& index
    )
    {
        while (index < edges.size() && key_at(edges, index) == key) {
            ++index;
        }
    }

    static std::array<std::uint64_t, 3> face_edge_keys(Triangle face)
    {
        return {
            edge_key(face.v0, face.v1),
            edge_key(face.v1, face.v2),
            edge_key(face.v2, face.v0),
        };
    }

    static bool is_duplicate(
        const KeyedTriangle& item,
        std::array<int, 3>& previous_key,
        bool& has_previous
    )
    {
        if (has_previous && item.key == previous_key) {
            return true;
        }
        has_previous = true;
        previous_key = item.key;
        return false;
    }

    std::vector<Triangle> select_with_sorted_edges() const
    {
        const std::vector<std::uint64_t> boundary_keys = merge_boundary_edge_keys();
        std::unordered_map<std::uint64_t, int> accepted_edge_counts;
        accepted_edge_counts.reserve(candidates_.size() * 3);
        std::vector<Triangle> out;
        out.reserve(keyed_candidates_.size());
        std::array<int, 3> previous_face_key{};
        bool has_previous_face = false;
        for (const KeyedTriangle& item : keyed_candidates_) {
            if (is_duplicate(item, previous_face_key, has_previous_face)) {
                continue;
            }
            const auto keys = face_edge_keys(item.face);
            if (sorted_face_is_manifold(keys, boundary_keys, accepted_edge_counts)) {
                out.push_back(item.face);
                increment_edge_counts(keys, accepted_edge_counts);
            }
        }
        return out;
    }

    static bool sorted_face_is_manifold(
        const std::array<std::uint64_t, 3>& face_keys,
        const std::vector<std::uint64_t>& boundary_keys,
        const std::unordered_map<std::uint64_t, int>& accepted_counts
    )
    {
        for (const std::uint64_t key : face_keys) {
            const auto found = accepted_counts.find(key);
            const int accepted_count = found == accepted_counts.end() ? 0 : found->second;
            const int boundary_count = std::binary_search(boundary_keys.begin(), boundary_keys.end(), key) ? 1 : 0;
            if (accepted_count + boundary_count >= 2) {
                return false;
            }
        }
        return true;
    }

    static void increment_edge_counts(
        const std::array<std::uint64_t, 3>& keys,
        std::unordered_map<std::uint64_t, int>& counts
    )
    {
        for (const std::uint64_t key : keys) {
            const auto [iter, inserted] = counts.try_emplace(key, 0);
            ++iter->second;
        }
    }

    std::unordered_map<std::uint64_t, int> initial_edge_counts() const
    {
        std::unordered_map<std::uint64_t, int> counts;
        counts.reserve(old_boundary_edges_.size() + new_boundary_edges_.size() + candidates_.size() * 3);
        for (const Edge edge : old_boundary_edges_) {
            counts[edge_key(edge[0], edge[1])] = 1;
        }
        for (const Edge edge : new_boundary_edges_) {
            counts[edge_key(edge[0], edge[1])] = 1;
        }
        return counts;
    }

    std::vector<Triangle> select_with_edge_map() const
    {
        std::unordered_map<std::uint64_t, int> edge_counts = initial_edge_counts();
        std::vector<Triangle> out;
        out.reserve(keyed_candidates_.size());
        std::array<int, 3> previous_face_key{};
        bool has_previous_face = false;
        for (const KeyedTriangle& item : keyed_candidates_) {
            if (is_duplicate(item, previous_face_key, has_previous_face)) {
                continue;
            }
            const auto keys = face_edge_keys(item.face);
            if (mapped_face_is_manifold(keys, edge_counts)) {
                out.push_back(item.face);
                increment_edge_counts(keys, edge_counts);
            }
        }
        return out;
    }

    static bool mapped_face_is_manifold(
        const std::array<std::uint64_t, 3>& keys,
        const std::unordered_map<std::uint64_t, int>& counts
    )
    {
        for (const std::uint64_t key : keys) {
            const auto found = counts.find(key);
            if (found != counts.end() && found->second >= 2) {
                return false;
            }
        }
        return true;
    }

    std::vector<Triangle> candidates_;
    const std::vector<Edge>& old_boundary_edges_;
    const std::vector<Edge>& new_boundary_edges_;
    std::vector<KeyedTriangle> keyed_candidates_;
};

std::vector<Triangle> select_manifold_faces(
    std::vector<Triangle> candidates,
    const std::vector<Edge>& old_boundary_edges,
    const std::vector<Edge>& new_boundary_edges
)
{
    return ManifoldFaceSelector(
        std::move(candidates), old_boundary_edges, new_boundary_edges
    ).run();
}


}  // namespace

namespace {

void add_timing(PipelineTiming* timing, const char* name, const Stopwatch& stopwatch)
{
    if (timing != nullptr) {
        timing->add_stage(name, stopwatch.elapsed_milliseconds());
    }
}

class ManifoldSelector {
public:
    ManifoldSelector(
        ManifoldSelectionRequest request,
        const SparseMeshingConfig& config,
        StitchCoreDebug* debug,
        PipelineTiming* timing
    )
        : request_(std::move(request)), config_(config), debug_(debug), timing_(timing)
    {
    }

    ManifoldSelectionResult run()
    {
        filter_candidates();
        select_manifold();
        orient_faces();
        return ManifoldSelectionResult{std::move(faces_), candidate_count_};
    }

private:
    void filter_candidates()
    {
        Stopwatch stopwatch;
        const MeshScalar max_edge_sq = config_.max_triangle_dist * config_.max_triangle_dist;
        request_.candidates.erase(
            std::remove_if(
                request_.candidates.begin(),
                request_.candidates.end(),
                [&](Triangle face) {
                    if (request_.connection_oracle != nullptr
                        && !request_.connection_oracle->face_allowed(
                            face, request_.old_vertex_count)) {
                        return true;
                    }
                    // Stitching can legitimately contain narrow triangles
                    // when old and new boundaries have different sampling.
                    // Standalone-mesh area and quality thresholds do not apply
                    // here; connection and distance checks still protect the
                    // stitching surface.
                    return !face_within_old_new_distance(
                               face,
                               request_.vertices,
                               request_.old_vertex_count,
                               config_.stitch_max_triangle_dist
                           )
                        || longest_edge_squared(
                               request_.vertices[face.v0],
                               request_.vertices[face.v1],
                               request_.vertices[face.v2]
                           ) > max_edge_sq;
                }
            ),
            request_.candidates.end()
        );
        candidate_count_ = request_.candidates.size();
        if (debug_ != nullptr) {
            debug_->candidate_faces_after_filter = request_.candidates;
        }
        add_timing(timing_, "stitch.core.filter_candidates", stopwatch);
    }

    void select_manifold()
    {
        Stopwatch stopwatch;
        faces_ = select_manifold_faces(
            std::move(request_.candidates),
            request_.old_boundary_edges,
            request_.new_boundary_edges
        );
        add_timing(timing_, "stitch.core.select_manifold", stopwatch);
    }

    void orient_faces()
    {
        Stopwatch stopwatch;
        for (Triangle& face : faces_) {
            orient_face_toward_sensor(face, request_.vertices);
        }
        if (debug_ != nullptr) {
            debug_->selected_faces_local = faces_;
        }
        add_timing(timing_, "stitch.core.orient_faces", stopwatch);
    }

    ManifoldSelectionRequest request_;
    const SparseMeshingConfig& config_;
    StitchCoreDebug* debug_{};
    PipelineTiming* timing_{};
    std::vector<Triangle> faces_;
    std::size_t candidate_count_{};
};

}  // namespace

ManifoldSelectionResult select_stitch_faces(
    ManifoldSelectionRequest request,
    const SparseMeshingConfig& config,
    StitchCoreDebug* debug,
    PipelineTiming* timing
)
{
    return ManifoldSelector(std::move(request), config, debug, timing).run();
}

}  // namespace ri_meshing::stitch_detail
