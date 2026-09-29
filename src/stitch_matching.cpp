#include "stitch_core_components.hpp"

#include "stitch_boundary_filters.hpp"

#include "ri_meshing/timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

#if defined(RI_MESHING_HAS_OPENMP)
#include <omp.h>
#endif

namespace ri_meshing::stitch_detail {
namespace {

constexpr MeshScalar kPi = static_cast<MeshScalar>(3.14159265358979323846264338327950288);

struct CellKey {
    int x{};
    int y{};
    int z{};

    bool operator==(const CellKey& other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CellKeyHash {
    std::size_t operator()(CellKey key) const noexcept
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

struct NewLabelEntry {
    int vertex_id{};
    int cluster{};
    int label{};
    MeshScalar path_t{};
    int row{};
    int col{};
};
struct OldLabelAssignmentInput {
    const std::vector<Vec3f>& vertices;
    const std::vector<BoundaryPath>& old_paths;
    const std::vector<BoundaryPath>& new_paths;
    int old_vertex_count{};
    bool old_ids_are_dense{};
    const StitchConnectionOracle* connection_oracle{};
};

struct OldVertexQuery {
    int vertex_id{};
    int row{};
    int col{};
    CellKey cell{};
};

void add_timing(PipelineTiming* timing, const char* name, const Stopwatch& stopwatch)
{
    if (timing != nullptr) {
        timing->add_stage(name, stopwatch.elapsed_milliseconds());
    }
}

MeshScalar squared_distance(Vec3f a, Vec3f b)
{
    const MeshScalar dx = a.x - b.x;
    const MeshScalar dy = a.y - b.y;
    const MeshScalar dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

std::array<int, 2> project_to_ri_pixel(Vec3f point, const SparseMeshingConfig& config)
{
    const MeshScalar x = point.x;
    const MeshScalar y = point.y;
    const MeshScalar z = point.z;
    const MeshScalar xy = std::hypot(x, y);
    const MeshScalar phi = std::atan2(z, xy);
    const MeshScalar theta = std::atan2(y, x);
    const MeshScalar fov_range = config.vertical_fov.max_rad - config.vertical_fov.min_rad;
    const int row = static_cast<int>(std::llround(
        (phi - config.vertical_fov.min_rad) / fov_range * static_cast<MeshScalar>(config.ri_height)
    ));
    int col = static_cast<int>(std::llround(
        (theta + kPi) / (static_cast<MeshScalar>(2) * kPi) * static_cast<MeshScalar>(config.ri_width)
    ));
    col = std::clamp(col, 0, static_cast<int>(config.ri_width) - 1);
    return {row, col};
}

CellKey point_cell(Vec3f point, MeshScalar cell_size)
{
    return CellKey{
        static_cast<int>(std::floor(static_cast<MeshScalar>(point.x) / cell_size)),
        static_cast<int>(std::floor(static_cast<MeshScalar>(point.y) / cell_size)),
        static_cast<int>(std::floor(static_cast<MeshScalar>(point.z) / cell_size)),
    };
}

class OldLabelAssigner {
public:
    OldLabelAssigner(
        const OldLabelAssignmentInput& input,
        const SparseMeshingConfig& config,
        const BoundaryIncidenceData& old_incidence,
        const BoundaryIncidenceData& new_incidence,
        PipelineTiming* timing
    )
        : input_(input), config_(config), old_incidence_(old_incidence), new_incidence_(new_incidence),
          cell_size_(config.boundary_label_max_euclidean_dist),
          max_dist_sq_(cell_size_ * cell_size_), width_(static_cast<int>(config.ri_width)),
          half_width_(width_ / 2), use_dense_old_ids_(dense_old_ids_are_usable()), timing_(timing),
          hits_(result_size())
    {
    }

    std::vector<LabelHit> run()
    {
        Stopwatch stopwatch;
        build_spatial_labels();
        add_timing(timing_, "stitch.core.assign_labels.build_spatial", stopwatch);
        stopwatch = Stopwatch{};
        collect_sparse_old_ids();
        add_timing(timing_, "stitch.core.assign_labels.collect_old_ids", stopwatch);
        stopwatch = Stopwatch{};
        assign_labels();
        add_timing(timing_, "stitch.core.assign_labels.assign_vertices", stopwatch);
        return std::move(hits_);
    }

private:
    bool dense_old_ids_are_usable() const
    {
        return input_.old_ids_are_dense && input_.old_vertex_count > 0
            && static_cast<std::size_t>(input_.old_vertex_count) <= input_.vertices.size();
    }

    std::size_t result_size() const
    {
        return input_.old_vertex_count > 0
                && static_cast<std::size_t>(input_.old_vertex_count) <= input_.vertices.size()
            ? static_cast<std::size_t>(input_.old_vertex_count)
            : input_.vertices.size();
    }

    std::array<int, 2> project_vertex(int vertex_id) const
    {
        if (input_.connection_oracle != nullptr) {
            return input_.connection_oracle->projected_pixel(vertex_id);
        }
        return project_to_ri_pixel(input_.vertices[vertex_id], config_);
    }

    void build_spatial_labels()
    {
        spatial_labels_.reserve(path_vertex_count(input_.new_paths));
        int next_global_label = 0;
        for (std::size_t cluster = 0; cluster < input_.new_paths.size(); ++cluster) {
            const BoundaryPath& path = input_.new_paths[cluster];
            const MeshScalar denom = path.ids.size() > 1
                ? static_cast<MeshScalar>(path.ids.size() - 1) : static_cast<MeshScalar>(1);
            for (std::size_t idx = 0; idx < path.ids.size(); ++idx) {
                const int vertex_id = path.ids[idx];
                const auto [row, col] = project_vertex(vertex_id);
                if (row < 0 || row >= static_cast<int>(config_.ri_height)) {
                    continue;
                }
                spatial_labels_[point_cell(input_.vertices[vertex_id], cell_size_)].push_back(NewLabelEntry{
                    vertex_id, static_cast<int>(cluster), next_global_label,
                    static_cast<MeshScalar>(idx) / denom, row, col,
                });
                ++next_global_label;
            }
        }
    }

    static std::size_t path_vertex_count(const std::vector<BoundaryPath>& paths)
    {
        return std::accumulate(
            paths.begin(), paths.end(), std::size_t{0},
            [](std::size_t sum, const BoundaryPath& path) { return sum + path.ids.size(); }
        );
    }

    void collect_sparse_old_ids()
    {
        if (use_dense_old_ids_) {
            return;
        }
        old_ids_to_label_.reserve(path_vertex_count(input_.old_paths));
        for (const BoundaryPath& path : input_.old_paths) {
            old_ids_to_label_.insert(old_ids_to_label_.end(), path.ids.begin(), path.ids.end());
        }
        std::sort(old_ids_to_label_.begin(), old_ids_to_label_.end());
        old_ids_to_label_.erase(
            std::unique(old_ids_to_label_.begin(), old_ids_to_label_.end()), old_ids_to_label_.end()
        );
    }

    void assign_labels()
    {
        const int old_count = use_dense_old_ids_
            ? input_.old_vertex_count
            : static_cast<int>(old_ids_to_label_.size());
#if defined(RI_MESHING_HAS_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int old_pos = 0; old_pos < old_count; ++old_pos) {
            assign_vertex(old_pos);
        }
    }

    void assign_vertex(int old_pos)
    {
        const int old_id = use_dense_old_ids_ ? old_pos : old_ids_to_label_[static_cast<std::size_t>(old_pos)];
        const auto [row, col] = project_vertex(old_id);
        if (row < 0 || row >= static_cast<int>(config_.ri_height)) {
            return;
        }
        const OldVertexQuery query{old_id, row, col, point_cell(input_.vertices[old_id], cell_size_)};
        hits_[static_cast<std::size_t>(old_id)] = find_best_label(query);
    }

    LabelHit find_best_label(const OldVertexQuery& query) const
    {
        LabelHit best;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto found = spatial_labels_.find(
                        CellKey{query.cell.x + dx, query.cell.y + dy, query.cell.z + dz}
                    );
                    if (found == spatial_labels_.end()) {
                        continue;
                    }
                    for (const NewLabelEntry& candidate : found->second) {
                        consider_candidate(query, candidate, best);
                    }
                }
            }
        }
        return best;
    }

    void consider_candidate(
        const OldVertexQuery& query,
        const NewLabelEntry& candidate,
        LabelHit& best
    ) const
    {
        constexpr int search_radius = 16;
        const int dr = candidate.row - query.row;
        int dc = candidate.col - query.col;
        if (dc > half_width_) {
            dc -= width_;
        } else if (dc < -half_width_) {
            dc += width_;
        }
        if (std::abs(dr) > search_radius || std::abs(dc) > search_radius) {
            return;
        }
        const MeshScalar geom_dist_sq = squared_distance(
            input_.vertices[query.vertex_id], input_.vertices[candidate.vertex_id]
        );
        if (geom_dist_sq > max_dist_sq_) {
            return;
        }
        // The incidence-level direction test is a cheap local rejection. Run
        // it before the component/path walk, which may inspect several RI
        // samples and red-component steps. Both predicates are pure filters,
        // so this changes evaluation order only, not accepted labels.
        if (!has_open_side_match(
                query.vertex_id,
                candidate.vertex_id,
                input_.vertices,
                old_incidence_,
                new_incidence_)) {
            return;
        }
        int component = -1;
        if (input_.connection_oracle != nullptr) {
            component = input_.connection_oracle->component_for_pair(
                query.vertex_id, candidate.vertex_id
            );
            if (component < 0) {
                return;
            }
        }
        const MeshScalar score = static_cast<MeshScalar>(dr * dr + dc * dc)
            + static_cast<MeshScalar>(1000) * geom_dist_sq;
        if (score < best.score) {
            best = LabelHit{candidate.cluster, candidate.label, candidate.path_t, score, component};
        }
    }

    const OldLabelAssignmentInput& input_;
    const SparseMeshingConfig& config_;
    const BoundaryIncidenceData& old_incidence_;
    const BoundaryIncidenceData& new_incidence_;
    MeshScalar cell_size_{};
    MeshScalar max_dist_sq_{};
    int width_{};
    int half_width_{};
    bool use_dense_old_ids_{};
    PipelineTiming* timing_{};
    std::vector<LabelHit> hits_;
    std::unordered_map<CellKey, std::vector<NewLabelEntry>, CellKeyHash> spatial_labels_;
    std::vector<int> old_ids_to_label_;
};

std::vector<LabelHit> assign_old_labels(
    const OldLabelAssignmentInput& input,
    const SparseMeshingConfig& config,
    const BoundaryIncidenceData& old_incidence,
    const BoundaryIncidenceData& new_incidence,
    PipelineTiming* timing
)
{
    return OldLabelAssigner(input, config, old_incidence, new_incidence, timing).run();
}

bool valid_label_hit(const LabelHit& hit)
{
    return hit.cluster >= 0 && hit.component >= 0 && std::isfinite(hit.path_t);
}

constexpr std::size_t kMaxRepairGapVertices = 8;

bool valid_label_at(
    const BoundaryPath& path,
    std::size_t position,
    const std::vector<LabelHit>& labels
)
{
    const int vertex = path.ids[position];
    return vertex >= 0 && static_cast<std::size_t>(vertex) < labels.size()
        && valid_label_hit(labels[static_cast<std::size_t>(vertex)]);
}

MeshScalar directed_path_t_delta(MeshScalar previous, MeshScalar next, bool closed)
{
    MeshScalar delta = next - previous;
    if (!closed) {
        return delta;
    }
    while (delta > 0.5) {
        delta -= 1.0;
    }
    while (delta < -0.5) {
        delta += 1.0;
    }
    return delta;
}

MeshScalar interpolate_path_t(MeshScalar start, MeshScalar delta, MeshScalar fraction, bool closed)
{
    MeshScalar value = start + fraction * delta;
    if (!closed) {
        return value;
    }
    while (value < 0.0) {
        value += 1.0;
    }
    while (value > 1.0) {
        value -= 1.0;
    }
    return value;
}

struct GapRun {
    std::vector<std::size_t> positions;
    std::size_t previous_position{};
    std::size_t next_position{};
};

bool repair_gap_run(
    const BoundaryPath& path,
    const GapRun& gap_run,
    std::vector<LabelHit>& labels,
    StitchConnectionOracle& connection_oracle
)
{
    if (gap_run.positions.empty() || gap_run.positions.size() > kMaxRepairGapVertices
        || !valid_label_at(path, gap_run.previous_position, labels)
        || !valid_label_at(path, gap_run.next_position, labels)) {
        return false;
    }
    const LabelHit& previous = labels[
        static_cast<std::size_t>(path.ids[gap_run.previous_position])
    ];
    const LabelHit& next = labels[
        static_cast<std::size_t>(path.ids[gap_run.next_position])
    ];
    if (previous.cluster != next.cluster || previous.component != next.component) {
        return false;
    }
    const MeshScalar path_delta = directed_path_t_delta(
        previous.path_t, next.path_t, path.closed
    );
    if (std::abs(path_delta) > 0.5) {
        return false;
    }

    const MeshScalar label_delta = next.label - previous.label;
    const MeshScalar count = static_cast<MeshScalar>(gap_run.positions.size() + 1);
    for (std::size_t index = 0; index < gap_run.positions.size(); ++index) {
        const std::size_t position = gap_run.positions[index];
        const int old_vertex = path.ids[position];
        if (old_vertex < 0 || static_cast<std::size_t>(old_vertex) >= labels.size()
            || valid_label_hit(labels[static_cast<std::size_t>(old_vertex)])) {
            return false;
        }
        LabelHit& gap = labels[static_cast<std::size_t>(old_vertex)];
        const MeshScalar fraction = static_cast<MeshScalar>(index + 1) / count;
        gap.cluster = previous.cluster;
        gap.label = static_cast<int>(std::llround(
            static_cast<MeshScalar>(previous.label) + fraction * label_delta
        ));
        gap.path_t = interpolate_path_t(previous.path_t, path_delta, fraction, path.closed);
        gap.score = std::max(previous.score, next.score);
        gap.component = previous.component;
        connection_oracle.inherit_component_support(old_vertex, gap.component);
    }
    return true;
}

void repair_short_gaps(
    const std::vector<BoundaryPath>& old_paths,
    std::vector<LabelHit>& labels,
    StitchConnectionOracle* connection_oracle
)
{
    if (connection_oracle == nullptr) {
        return;
    }
    for (const BoundaryPath& path : old_paths) {
        if (path.ids.size() < 3) {
            continue;
        }
        const std::size_t path_size = path.ids.size();
        std::size_t scan_start = 0;
        if (path.closed) {
            const auto first_valid = std::find_if(
                path.ids.begin(), path.ids.end(), [&](int vertex) {
                    return vertex >= 0 && static_cast<std::size_t>(vertex) < labels.size()
                        && valid_label_hit(labels[static_cast<std::size_t>(vertex)]);
                }
            );
            if (first_valid == path.ids.end()) {
                continue;
            }
            scan_start = static_cast<std::size_t>(first_valid - path.ids.begin());
        }

        std::size_t offset = 0;
        while (offset < path_size) {
            const std::size_t position = (scan_start + offset) % path_size;
            if (valid_label_at(path, position, labels)) {
                ++offset;
                continue;
            }
            const std::size_t run_begin = offset;
            while (offset < path_size
                   && !valid_label_at(path, (scan_start + offset) % path_size, labels)) {
                ++offset;
            }
            const std::size_t run_size = offset - run_begin;
            if (run_size == 0 || run_size > kMaxRepairGapVertices) {
                continue;
            }
            if (!path.closed && (run_begin == 0 || offset == path_size)) {
                continue;
            }

            std::vector<std::size_t> gap_positions;
            gap_positions.reserve(run_size);
            for (std::size_t index = 0; index < run_size; ++index) {
                gap_positions.push_back((scan_start + run_begin + index) % path_size);
            }
            const std::size_t previous_position =
                (gap_positions.front() + path_size - 1) % path_size;
            const std::size_t next_position = (gap_positions.back() + 1) % path_size;
            repair_gap_run(
                path,
                GapRun{
                    std::move(gap_positions), previous_position, next_position
                },
                labels, *connection_oracle
            );
        }
    }
}

std::vector<std::vector<OldLabelSegment>> old_label_segments_by_new_cluster(
    const std::vector<BoundaryPath>& old_paths,
    const std::vector<LabelHit>& old_labels,
    std::size_t new_path_count
)
{
    std::vector<std::vector<OldLabelSegment>> segments_by_cluster(new_path_count);
    for (std::size_t old_path_index = 0; old_path_index < old_paths.size(); ++old_path_index) {
        const BoundaryPath& old_path = old_paths[old_path_index];
        OldLabelSegment segment;
        int current_cluster = -1;
        int current_component = -1;
        const auto flush_segment = [&] {
            if (current_cluster < 0 || segment.ids.size() < 2) {
                segment.ids.clear();
                segment.labels.clear();
                segment.closed = false;
                segment.component = -1;
                return;
            }
            if (segment.labels.front() > segment.labels.back()) {
                std::reverse(segment.ids.begin(), segment.ids.end());
                std::reverse(segment.labels.begin(), segment.labels.end());
            }
            segment.closed = old_path.closed && segment.ids.size() == old_path.ids.size();
            segment.component = current_component;
            segment.source_path = static_cast<int>(old_path_index);
            segments_by_cluster[static_cast<std::size_t>(current_cluster)].push_back(std::move(segment));
            segment = OldLabelSegment{};
        };

        for (const int old_id : old_path.ids) {
            const LabelHit hit = old_labels[static_cast<std::size_t>(old_id)];
            const int next_cluster =
                hit.cluster >= 0 && static_cast<std::size_t>(hit.cluster) < new_path_count ? hit.cluster : -1;
            const int next_component = next_cluster >= 0 ? hit.component : -1;
            if (next_cluster != current_cluster || next_component != current_component) {
                flush_segment();
                current_cluster = next_cluster;
                current_component = next_component;
            }
            if (current_cluster >= 0) {
                segment.ids.push_back(old_id);
                segment.labels.push_back(hit.path_t);
            }
        }
        flush_segment();
    }
    return segments_by_cluster;
}

bool segment_contains_vertex(const OldLabelSegment& segment, int vertex)
{
    return std::find(segment.ids.begin(), segment.ids.end(), vertex) != segment.ids.end();
}

int nearest_new_path_position(const BoundaryPath& path, MeshScalar target)
{
    if (path.ids.empty()) {
        return -1;
    }
    const MeshScalar denominator = path.ids.size() > 1
        ? static_cast<MeshScalar>(path.ids.size() - 1) : static_cast<MeshScalar>(1);
    int best_position = -1;
    MeshScalar best_distance = std::numeric_limits<MeshScalar>::infinity();
    for (std::size_t position = 0; position < path.ids.size(); ++position) {
        const MeshScalar value = static_cast<MeshScalar>(position) / denominator;
        MeshScalar distance = std::abs(value - target);
        if (path.closed) {
            distance = std::min(distance, static_cast<MeshScalar>(1) - distance);
        }
        if (distance < best_distance) {
            best_distance = distance;
            best_position = static_cast<int>(position);
        }
    }
    return best_position;
}

bool path_positions_are_consecutive(
    const BoundaryPath& path,
    int first,
    int second
)
{
    if (first < 0 || second < 0 || first >= static_cast<int>(path.ids.size())
        || second >= static_cast<int>(path.ids.size())) {
        return false;
    }
    return second == first + 1
        || (path.closed && first == static_cast<int>(path.ids.size()) - 1 && second == 0);
}

bool merge_wrapped_segment_pair(
    const BoundaryPath& old_path,
    const BoundaryPath& new_path,
    std::vector<OldLabelSegment>& segments
)
{
    if (!old_path.closed || old_path.ids.size() < 3 || new_path.ids.size() < 2) {
        return false;
    }
    int first_index = -1;
    int last_index = -1;
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const OldLabelSegment& segment = segments[index];
        if (!segment_contains_vertex(segment, old_path.ids.front())
            && !segment_contains_vertex(segment, old_path.ids.back())) {
            continue;
        }
        if (segment_contains_vertex(segment, old_path.ids.front())) {
            first_index = static_cast<int>(index);
        }
        if (segment_contains_vertex(segment, old_path.ids.back())) {
            last_index = static_cast<int>(index);
        }
    }
    if (first_index < 0 || last_index < 0 || first_index == last_index) {
        return false;
    }

    const OldLabelSegment& first_segment = segments[static_cast<std::size_t>(first_index)];
    const OldLabelSegment& last_segment = segments[static_cast<std::size_t>(last_index)];
    if (first_segment.source_path != last_segment.source_path
        || first_segment.component != last_segment.component) {
        return false;
    }

    const bool first_then_last = first_segment.ids.back() == old_path.ids.front()
        && last_segment.ids.front() == old_path.ids.back();
    const bool last_then_first = last_segment.ids.back() == old_path.ids.back()
        && first_segment.ids.front() == old_path.ids.front();
    if (!first_then_last && !last_then_first) {
        return false;
    }

    const MeshScalar first_label = first_then_last
        ? first_segment.labels.back() : first_segment.labels.front();
    const MeshScalar last_label = first_then_last
        ? last_segment.labels.front() : last_segment.labels.back();
    const int first_new_position = nearest_new_path_position(new_path, first_label);
    const int last_new_position = nearest_new_path_position(new_path, last_label);
    const bool labels_follow_first_then_last = path_positions_are_consecutive(
        new_path, first_new_position, last_new_position
    );
    const bool labels_follow_last_then_first = path_positions_are_consecutive(
        new_path, last_new_position, first_new_position
    );
    if ((first_then_last && !labels_follow_first_then_last)
        || (last_then_first && !labels_follow_last_then_first)) {
        return false;
    }

    OldLabelSegment merged = first_then_last ? first_segment : last_segment;
    const OldLabelSegment& tail = first_then_last ? last_segment : first_segment;
    merged.ids.insert(merged.ids.end(), tail.ids.begin(), tail.ids.end());
    merged.labels.insert(merged.labels.end(), tail.labels.begin(), tail.labels.end());
    merged.closed = old_path.ids.size() == merged.ids.size();

    const int keep_index = std::min(first_index, last_index);
    const int remove_index = std::max(first_index, last_index);
    segments[static_cast<std::size_t>(keep_index)] = std::move(merged);
    segments.erase(segments.begin() + remove_index);
    return true;
}

void merge_wrapped_segments(
    std::vector<std::vector<OldLabelSegment>>& segments_by_cluster,
    const std::vector<BoundaryPath>& old_paths,
    const std::vector<BoundaryPath>& new_paths
)
{
    for (std::size_t cluster = 0; cluster < segments_by_cluster.size(); ++cluster) {
        if (cluster >= new_paths.size()) {
            continue;
        }
        std::vector<OldLabelSegment>& segments = segments_by_cluster[cluster];
        for (const BoundaryPath& old_path : old_paths) {
            merge_wrapped_segment_pair(old_path, new_paths[cluster], segments);
        }
    }
}


}  // namespace

namespace {

class BoundaryMatcher {
public:
    BoundaryMatcher(
        const BoundaryMatchingRequest& request,
        const SparseMeshingConfig& config,
        StitchCoreDebug* debug,
        PipelineTiming* timing
    )
        : request_(request), config_(config), debug_(debug), timing_(timing)
    {
    }

    BoundaryMatchingResult run()
    {
        build_incidences();
        Stopwatch stopwatch;
        if (request_.connection_oracle != nullptr) {
            stopwatch = Stopwatch{};
            annotate_open_side_component_pair(
                old_incidence_, new_incidence_, *request_.connection_oracle
            );
            add_timing(timing_, "stitch.core.incidence.annotate_open_side", stopwatch);
        }
        assign_labels();
        stopwatch = Stopwatch{};
        repair_short_gaps(request_.old_paths, labels_, request_.connection_oracle);
        add_timing(timing_, "stitch.core.repair_label_gaps", stopwatch);
        publish_label_debug();
        group_segments();
        return BoundaryMatchingResult{
            std::move(segments_),
            std::move(old_incidence_),
            std::move(new_incidence_),
        };
    }

private:
    void build_incidences()
    {
        double old_milliseconds = 0.0;
        double new_milliseconds = 0.0;
        const bool enough_work = request_.old_boundary_edges.size()
            + request_.new_boundary_edges.size() > 256;
#if defined(RI_MESHING_HAS_OPENMP)
        #pragma omp parallel sections if(enough_work)
        {
            #pragma omp section
            {
                const Stopwatch stopwatch;
                old_incidence_ = build_boundary_incidence(
                    request_.old_paths,
                    request_.old_boundary_edges,
                    request_.vertices,
                    request_.vertex_normals
                );
                old_milliseconds = stopwatch.elapsed_milliseconds();
            }
            #pragma omp section
            {
                const Stopwatch stopwatch;
                new_incidence_ = build_boundary_incidence(
                    request_.new_paths,
                    request_.new_boundary_edges,
                    request_.vertices,
                    request_.vertex_normals
                );
                new_milliseconds = stopwatch.elapsed_milliseconds();
            }
        }
#else
        {
            const Stopwatch stopwatch;
            old_incidence_ = build_boundary_incidence(
                request_.old_paths,
                request_.old_boundary_edges,
                request_.vertices,
                request_.vertex_normals
            );
            old_milliseconds = stopwatch.elapsed_milliseconds();
        }
        {
            const Stopwatch stopwatch;
            new_incidence_ = build_boundary_incidence(
                request_.new_paths,
                request_.new_boundary_edges,
                request_.vertices,
                request_.vertex_normals
            );
            new_milliseconds = stopwatch.elapsed_milliseconds();
        }
#endif
        if (timing_ != nullptr) {
            timing_->add_stage("stitch.core.incidence.old", old_milliseconds);
            timing_->add_stage("stitch.core.incidence.new", new_milliseconds);
        }
    }

    void assign_labels()
    {
        Stopwatch stopwatch;
        labels_ = assign_old_labels(
            OldLabelAssignmentInput{
                request_.vertices,
                request_.old_paths,
                request_.new_paths,
                request_.old_vertex_count,
                request_.old_ids_are_dense,
                request_.connection_oracle,
            },
            config_,
            old_incidence_,
            new_incidence_,
            timing_
        );
        add_timing(timing_, "stitch.core.assign_labels", stopwatch);
    }

    void publish_label_debug()
    {
        if (debug_ == nullptr) {
            return;
        }
        debug_->old_labels.clear();
        std::vector<int> old_ids;
        const std::size_t old_refs = std::accumulate(
            request_.old_paths.begin(),
            request_.old_paths.end(),
            std::size_t{0},
            [](std::size_t sum, const BoundaryPath& path) { return sum + path.ids.size(); }
        );
        old_ids.reserve(old_refs);
        for (const BoundaryPath& path : request_.old_paths) {
            old_ids.insert(old_ids.end(), path.ids.begin(), path.ids.end());
        }
        std::sort(old_ids.begin(), old_ids.end());
        old_ids.erase(std::unique(old_ids.begin(), old_ids.end()), old_ids.end());
        debug_->old_labels.reserve(old_ids.size());
        for (const int vertex_id : old_ids) {
            const LabelHit hit = labels_[static_cast<std::size_t>(vertex_id)];
            debug_->old_labels.push_back(StitchDebugLabel{
                vertex_id, hit.cluster, hit.label, hit.path_t, hit.score,
            });
        }
    }

    void group_segments()
    {
        Stopwatch stopwatch;
        segments_ = old_label_segments_by_new_cluster(
            request_.old_paths, labels_, request_.new_paths.size()
        );
        merge_wrapped_segments(segments_, request_.old_paths, request_.new_paths);
        publish_segment_debug();
        add_timing(timing_, "stitch.core.group_segments", stopwatch);
    }

    void publish_segment_debug()
    {
        if (debug_ == nullptr) {
            return;
        }
        debug_->old_segments.clear();
        for (std::size_t cluster = 0; cluster < segments_.size(); ++cluster) {
            const std::vector<OldLabelSegment>& segments = segments_[cluster];
            for (std::size_t segment = 0; segment < segments.size(); ++segment) {
                debug_->old_segments.push_back(StitchDebugSegment{
                    static_cast<int>(cluster),
                    static_cast<int>(segment),
                    segments[segment].ids,
                    segments[segment].labels,
                });
            }
        }
    }

    const BoundaryMatchingRequest& request_;
    const SparseMeshingConfig& config_;
    StitchCoreDebug* debug_{};
    PipelineTiming* timing_{};
    std::vector<LabelHit> labels_;
    std::vector<std::vector<OldLabelSegment>> segments_;
    BoundaryIncidenceData old_incidence_;
    BoundaryIncidenceData new_incidence_;
};

}  // namespace

BoundaryMatchingResult match_boundaries(
    const BoundaryMatchingRequest& request,
    const SparseMeshingConfig& config,
    StitchCoreDebug* debug,
    PipelineTiming* timing
)
{
    return BoundaryMatcher(request, config, debug, timing).run();
}

}  // namespace ri_meshing::stitch_detail
