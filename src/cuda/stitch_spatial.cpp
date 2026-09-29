#include "stitch_spatial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ri_meshing::cuda_pipeline {

void sorted_unique_vertex_ids(
    const std::vector<std::array<int, 2>>& edges,
    int old_global_vertex_count,
    bool want_old,
    std::vector<int>& ids,
    PipelineTiming* timing = nullptr
)
{
    Stopwatch stopwatch;
    ids.clear();
    ids.reserve(edges.size() * 2);
    for (const std::array<int, 2> edge : edges) {
        for (const int id : edge) {
            if (id < 0) {
                continue;
            }
            const bool is_old = id < old_global_vertex_count;
            if (is_old == want_old) {
                ids.push_back(id);
            }
        }
    }
    if (timing != nullptr) {
        timing->add_stage("stitch.compact.collect_ids.scan_edges", stopwatch.elapsed_milliseconds());
    }

    stopwatch = Stopwatch{};
    std::sort(ids.begin(), ids.end());
    if (timing != nullptr) {
        timing->add_stage("stitch.compact.collect_ids.sort", stopwatch.elapsed_milliseconds());
    }

    stopwatch = Stopwatch{};
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (timing != nullptr) {
        timing->add_stage("stitch.compact.collect_ids.unique", stopwatch.elapsed_milliseconds());
    }
}

MeshScalar stitch_spatial_filter_radius(const SparseMeshingConfig& config)
{
    return std::max(config.boundary_label_max_euclidean_dist, config.stitch_max_triangle_dist)
        + config.max_triangle_dist;
}

StitchSpatialKey stitch_spatial_key(Vec3f point, MeshScalar cell_size)
{
    return StitchSpatialKey{
        static_cast<int>(std::floor(static_cast<MeshScalar>(point.x) / cell_size)),
        static_cast<int>(std::floor(static_cast<MeshScalar>(point.y) / cell_size)),
        static_cast<int>(std::floor(static_cast<MeshScalar>(point.z) / cell_size)),
    };
}

void ensure_spatial_boundary_index_capacity(StitchSpatialBoundaryIndex& index, std::size_t vertex_count)
{
    if (index.vertex_cell_keys.size() < vertex_count) {
        if (index.vertex_cell_keys.capacity() < vertex_count) {
            const std::size_t grown_capacity = std::max(vertex_count, index.vertex_cell_keys.capacity() * 3 / 2 + 1024);
            index.vertex_cell_keys.reserve(grown_capacity);
            index.vertex_cell_positions.reserve(grown_capacity);
            index.active_boundary_flags.reserve(grown_capacity);
            index.near_old_vertex_flags.reserve(grown_capacity);
            index.filtered_old_vertex_flags.reserve(grown_capacity);
        }
        index.vertex_cell_keys.resize(vertex_count);
        index.vertex_cell_positions.resize(vertex_count, std::numeric_limits<std::size_t>::max());
        index.active_boundary_flags.resize(vertex_count, std::uint8_t{0});
        index.near_old_vertex_flags.resize(vertex_count, std::uint8_t{0});
        index.filtered_old_vertex_flags.resize(vertex_count, std::uint8_t{0});
    }
}

void add_active_boundary_vertex(
    StitchSpatialBoundaryIndex& index,
    const std::vector<Vec3f>& vertices_global,
    int vertex_id
)
{
    const std::size_t id = static_cast<std::size_t>(vertex_id);
    if (index.active_boundary_flags[id] != 0) {
        return;
    }
    const StitchSpatialKey key = stitch_spatial_key(vertices_global[id], index.cell_size);
    std::vector<int>& cell = index.active_cells[key];
    index.vertex_cell_keys[id] = key;
    index.vertex_cell_positions[id] = cell.size();
    index.active_boundary_flags[id] = 1;
    cell.push_back(vertex_id);
}

void remove_active_boundary_vertex(StitchSpatialBoundaryIndex& index, int vertex_id)
{
    const std::size_t id = static_cast<std::size_t>(vertex_id);
    if (id >= index.active_boundary_flags.size() || index.active_boundary_flags[id] == 0) {
        return;
    }
    auto cell_iter = index.active_cells.find(index.vertex_cell_keys[id]);
    if (cell_iter != index.active_cells.end()) {
        std::vector<int>& cell = cell_iter->second;
        std::size_t pos = index.vertex_cell_positions[id];
        if (pos >= cell.size() || cell[pos] != vertex_id) {
            const auto found = std::find(cell.begin(), cell.end(), vertex_id);
            pos = found == cell.end() ? cell.size() : static_cast<std::size_t>(found - cell.begin());
        }
        if (pos < cell.size()) {
            const int moved_id = cell.back();
            cell[pos] = moved_id;
            index.vertex_cell_positions[static_cast<std::size_t>(moved_id)] = pos;
            cell.pop_back();
        }
    }
    index.active_boundary_flags[id] = 0;
    index.vertex_cell_positions[id] = std::numeric_limits<std::size_t>::max();
}

void update_active_boundary_index(
    StitchSpatialBoundaryIndex& index,
    const std::vector<Vec3f>& vertices_global,
    const std::vector<int>& old_boundary_vertex_ids,
    MeshScalar cell_size
)
{
    if (index.cell_size != cell_size) {
        index.clear();
        index.cell_size = cell_size;
    }
    ensure_spatial_boundary_index_capacity(index, vertices_global.size());
    index.active_cells.reserve(index.active_cells.size() + old_boundary_vertex_ids.size() / 8 + 1);

    std::size_t previous_idx = 0;
    std::size_t current_idx = 0;
    while (previous_idx < index.active_boundary_ids.size() || current_idx < old_boundary_vertex_ids.size()) {
        if (previous_idx >= index.active_boundary_ids.size()) {
            const int id = old_boundary_vertex_ids[current_idx++];
            if (id >= 0 && static_cast<std::size_t>(id) < vertices_global.size()) {
                add_active_boundary_vertex(index, vertices_global, id);
            }
            continue;
        }
        if (current_idx >= old_boundary_vertex_ids.size()) {
            remove_active_boundary_vertex(index, index.active_boundary_ids[previous_idx++]);
            continue;
        }
        const int previous_id = index.active_boundary_ids[previous_idx];
        const int current_id = old_boundary_vertex_ids[current_idx];
        if (previous_id == current_id) {
            ++previous_idx;
            ++current_idx;
        } else if (previous_id < current_id) {
            remove_active_boundary_vertex(index, previous_id);
            ++previous_idx;
        } else {
            if (current_id >= 0 && static_cast<std::size_t>(current_id) < vertices_global.size()) {
                add_active_boundary_vertex(index, vertices_global, current_id);
            }
            ++current_idx;
        }
    }
    index.active_boundary_ids = old_boundary_vertex_ids;
    index.initialized = true;
}

void apply_active_boundary_changes(
    StitchSpatialBoundaryIndex& index,
    const std::vector<Vec3f>& vertices_global,
    const std::vector<int>& current_boundary_vertex_ids,
    const std::vector<int>& changed_vertex_ids,
    MeshScalar cell_size
)
{
    if (index.cell_size != cell_size) {
        index.clear();
        index.cell_size = cell_size;
    }
    ensure_spatial_boundary_index_capacity(index, vertices_global.size());
    if (!index.initialized) {
        index.initialized = true;
        if (changed_vertex_ids.empty() && !current_boundary_vertex_ids.empty()) {
            update_active_boundary_index(index, vertices_global, current_boundary_vertex_ids, cell_size);
            return;
        }
    } else if (changed_vertex_ids.empty()) {
        return;
    }
    index.active_cells.reserve(index.active_cells.size() + changed_vertex_ids.size() / 4 + 1);
    for (const int vertex_id : changed_vertex_ids) {
        if (vertex_id < 0 || static_cast<std::size_t>(vertex_id) >= vertices_global.size()) {
            continue;
        }
        if (index.active_boundary_flags[static_cast<std::size_t>(vertex_id)] != 0) {
            remove_active_boundary_vertex(index, vertex_id);
        } else {
            add_active_boundary_vertex(index, vertices_global, vertex_id);
        }
    }
}

void build_new_boundary_index(
    StitchSpatialBoundaryIndex& spatial_index,
    const std::vector<Vec3f>& vertices_global,
    const std::vector<int>& new_ids,
    MeshScalar radius
)
{
    spatial_index.new_cells.clear();
    spatial_index.new_cells.reserve(new_ids.size());
    for (const int new_id : new_ids) {
        if (new_id < 0 || static_cast<std::size_t>(new_id) >= vertices_global.size()) {
            continue;
        }
        spatial_index.new_cells[stitch_spatial_key(vertices_global[static_cast<std::size_t>(new_id)], radius)]
            .push_back(new_id);
    }
}

void mark_near_old_boundary_vertices(
    StitchSpatialBoundaryIndex& spatial_index,
    const SpatialBoundaryFilterRequest& request,
    MeshScalar radius_sq,
    std::size_t new_id_count
)
{
    spatial_index.near_old_vertex_ids.clear();
    spatial_index.near_old_vertex_ids.reserve(
        std::min(request.old_boundary_vertex_ids.size(), new_id_count * 4)
    );
    for (const auto& [cell_key, new_cell_ids] : spatial_index.new_cells) {
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto found = spatial_index.active_cells.find(
                        StitchSpatialKey{cell_key.x + dx, cell_key.y + dy, cell_key.z + dz}
                    );
                    if (found == spatial_index.active_cells.end()) {
                        continue;
                    }
                    for (const int old_id : found->second) {
                        if (old_id >= request.old_global_vertex_count
                            || spatial_index.near_old_vertex_flags[static_cast<std::size_t>(old_id)] != 0) {
                            continue;
                        }
                        const Vec3f old_point = request.vertices_global[static_cast<std::size_t>(old_id)];
                        for (const int new_id : new_cell_ids) {
                            const Vec3f new_point = request.vertices_global[static_cast<std::size_t>(new_id)];
                            const MeshScalar dxp = new_point.x - old_point.x;
                            const MeshScalar dyp = new_point.y - old_point.y;
                            const MeshScalar dzp = new_point.z - old_point.z;
                            if (dxp * dxp + dyp * dyp + dzp * dzp <= radius_sq) {
                                spatial_index.near_old_vertex_flags[static_cast<std::size_t>(old_id)] = 1;
                                spatial_index.near_old_vertex_ids.push_back(old_id);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

void collect_filtered_old_boundary(
    const SpatialBoundaryFilterRequest& request,
    StitchSpatialBoundaryIndex& spatial_index,
    FilteredOldBoundaryEdges& result
)
{
    result.edges.clear();
    result.edges.reserve(std::min(
        request.old_boundary_edge_count,
        result.new_vertex_ids.size() * 8
    ));
    spatial_index.filtered_old_vertex_ids.clear();
    spatial_index.filtered_old_vertex_ids.reserve(
        std::min(request.old_boundary_vertex_ids.size(), result.new_vertex_ids.size() * 8)
    );
    for (const ri_meshing::BoundaryGraph::EdgeBlock& block : request.old_boundary_edge_blocks) {
        for (const std::array<int, 2> edge : block) {
            const bool keep_a = edge[0] >= 0
                && static_cast<std::size_t>(edge[0]) < spatial_index.near_old_vertex_flags.size()
                && spatial_index.near_old_vertex_flags[static_cast<std::size_t>(edge[0])] != 0;
            const bool keep_b = edge[1] >= 0
                && static_cast<std::size_t>(edge[1]) < spatial_index.near_old_vertex_flags.size()
                && spatial_index.near_old_vertex_flags[static_cast<std::size_t>(edge[1])] != 0;
            if (keep_a || keep_b) {
                result.edges.push_back(edge);
                for (const int vertex_id : edge) {
                    if (vertex_id >= 0 && vertex_id < request.old_global_vertex_count) {
                        const auto id = static_cast<std::size_t>(vertex_id);
                        if (spatial_index.filtered_old_vertex_flags[id] == 0) {
                            spatial_index.filtered_old_vertex_flags[id] = 1;
                            spatial_index.filtered_old_vertex_ids.push_back(vertex_id);
                        }
                    }
                }
            }
        }
    }
    std::sort(
        spatial_index.filtered_old_vertex_ids.begin(),
        spatial_index.filtered_old_vertex_ids.end()
    );
    for (const int vertex_id : spatial_index.near_old_vertex_ids) {
        spatial_index.near_old_vertex_flags[static_cast<std::size_t>(vertex_id)] = 0;
    }
    result.old_vertex_ids = spatial_index.filtered_old_vertex_ids;
    for (const int vertex_id : spatial_index.filtered_old_vertex_ids) {
        spatial_index.filtered_old_vertex_flags[static_cast<std::size_t>(vertex_id)] = 0;
    }
}

void filter_old_boundary_edges_near_new(
    const SpatialBoundaryFilterRequest& request,
    StitchSpatialBoundaryIndex& spatial_index,
    FilteredOldBoundaryEdges& result,
    PipelineTiming* timing
)
{
    Stopwatch stopwatch;
    result.edges.clear();
    result.old_vertex_ids.clear();
    result.new_vertex_ids.clear();
    sorted_unique_vertex_ids(
        request.new_boundary_edges, request.old_global_vertex_count, false,
        result.new_vertex_ids
    );
    if (result.new_vertex_ids.empty() || request.old_boundary_edge_blocks.empty()) {
        result.new_vertex_ids.clear();
        return;
    }

    const MeshScalar radius = stitch_spatial_filter_radius(request.config);
    const MeshScalar radius_sq = radius * radius;
    if (!spatial_index.initialized || spatial_index.cell_size != radius) {
        update_active_boundary_index(
            spatial_index,
            request.vertices_global,
            request.old_boundary_vertex_ids,
            radius
        );
    } else {
        ensure_spatial_boundary_index_capacity(spatial_index, request.vertices_global.size());
    }
    if (timing != nullptr) {
        timing->add_stage("stitch.compact.filter_old_boundary.update_old_index", stopwatch.elapsed_milliseconds());
    }

    stopwatch = Stopwatch{};
    build_new_boundary_index(spatial_index, request.vertices_global, result.new_vertex_ids, radius);
    if (timing != nullptr) {
        timing->add_stage("stitch.compact.filter_old_boundary.build_new_index", stopwatch.elapsed_milliseconds());
    }

    stopwatch = Stopwatch{};
    mark_near_old_boundary_vertices(
        spatial_index,
        request,
        radius_sq,
        result.new_vertex_ids.size()
    );
    if (timing != nullptr) {
        timing->add_stage("stitch.compact.filter_old_boundary.test_old_candidates", stopwatch.elapsed_milliseconds());
    }

    stopwatch = Stopwatch{};
    collect_filtered_old_boundary(request, spatial_index, result);
    if (timing != nullptr) {
        timing->add_stage("stitch.compact.filter_old_boundary.scan_old_edges", stopwatch.elapsed_milliseconds());
    }
}

}  // namespace ri_meshing::cuda_pipeline
