#include "stitch_boundary_filters.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ri_meshing::stitch_detail {
namespace {

constexpr MeshScalar kPi = static_cast<MeshScalar>(3.14159265358979323846264338327950288);

struct AngularPoint {
    MeshScalar azimuth{};
    MeshScalar elevation{};
};

struct BoundaryEdgeGeometry {
    Edge edge{};
    Vec3f tangent{};
    Vec2d angular_tangent{};
};

std::uint64_t edge_key(int a, int b)
{
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

Vec3f add(Vec3f a, Vec3f b)
{
    return Vec3f{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3f subtract(Vec3f a, Vec3f b)
{
    return Vec3f{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3f scale(Vec3f value, MeshScalar factor)
{
    return Vec3f{
        static_cast<float>(value.x * factor),
        static_cast<float>(value.y * factor),
        static_cast<float>(value.z * factor),
    };
}

MeshScalar dot(Vec3f a, Vec3f b)
{
    return static_cast<MeshScalar>(a.x) * b.x
        + static_cast<MeshScalar>(a.y) * b.y
        + static_cast<MeshScalar>(a.z) * b.z;
}

Vec3f cross(Vec3f a, Vec3f b)
{
    return Vec3f{
        static_cast<float>(static_cast<MeshScalar>(a.y) * b.z - static_cast<MeshScalar>(a.z) * b.y),
        static_cast<float>(static_cast<MeshScalar>(a.z) * b.x - static_cast<MeshScalar>(a.x) * b.z),
        static_cast<float>(static_cast<MeshScalar>(a.x) * b.y - static_cast<MeshScalar>(a.y) * b.x),
    };
}

MeshScalar squared_norm(Vec3f value)
{
    return dot(value, value);
}

MeshScalar dot(Vec2d a, Vec2d b)
{
    return a.x * b.x + a.y * b.y;
}

MeshScalar squared_norm(Vec2d value)
{
    return dot(value, value);
}

Vec2d normalized(Vec2d value)
{
    const MeshScalar length_sq = squared_norm(value);
    if (length_sq <= static_cast<MeshScalar>(1e-24)) {
        return Vec2d{};
    }
    const MeshScalar inverse_length = static_cast<MeshScalar>(1) / std::sqrt(length_sq);
    return Vec2d{value.x * inverse_length, value.y * inverse_length};
}

Vec3f normalized(Vec3f value)
{
    const MeshScalar length_sq = squared_norm(value);
    return length_sq > static_cast<MeshScalar>(1e-24)
        ? scale(value, static_cast<MeshScalar>(1) / std::sqrt(length_sq)) : Vec3f{};
}

AngularPoint angular_point(Vec3f point)
{
    const MeshScalar x = point.x;
    const MeshScalar y = point.y;
    const MeshScalar z = point.z;
    return AngularPoint{std::atan2(y, x), std::atan2(z, std::hypot(x, y))};
}

MeshScalar wrapped_azimuth_delta(MeshScalar start, MeshScalar end)
{
    return std::remainder(end - start, 2.0 * kPi);
}

Vec2d angular_edge_tangent(Edge edge, const std::vector<Vec3f>& vertices)
{
    const AngularPoint start = angular_point(vertices[static_cast<std::size_t>(edge[0])]);
    const AngularPoint end = angular_point(vertices[static_cast<std::size_t>(edge[1])]);
    const MeshScalar mean_elevation = static_cast<MeshScalar>(0.5) * (start.elevation + end.elevation);
    return normalized(Vec2d{
        wrapped_azimuth_delta(start.azimuth, end.azimuth) * std::cos(mean_elevation),
        end.elevation - start.elevation,
    });
}

Vec2d angular_direction(Vec3f start, Vec3f end)
{
    const AngularPoint start_point = angular_point(start);
    const AngularPoint end_point = angular_point(end);
    const MeshScalar mean_elevation = static_cast<MeshScalar>(0.5) * (start_point.elevation + end_point.elevation);
    return normalized(Vec2d{
        wrapped_azimuth_delta(start_point.azimuth, end_point.azimuth) * std::cos(mean_elevation),
        end_point.elevation - start_point.elevation,
    });
}

Vec2d angular_exterior_direction(Vec2d tangent)
{
    // Boundary edges preserve the oriented edge of a range-image CCW face.
    // Elevation grows upwards here, so the face is clockwise in this chart and
    // the exterior is the left-hand side of the oriented edge.
    return normalized(Vec2d{-tangent.y, tangent.x});
}

Vec3f normal_for_vertex(
    int vertex,
    const std::vector<Vec3f>& vertices,
    const std::vector<Vec3f>& vertex_normals
)
{
    if (vertex >= 0 && static_cast<std::size_t>(vertex) < vertex_normals.size()) {
        const Vec3f normal = normalized(vertex_normals[static_cast<std::size_t>(vertex)]);
        if (squared_norm(normal) > 0.0) {
            return normal;
        }
    }
    return vertex >= 0 && static_cast<std::size_t>(vertex) < vertices.size()
        ? normalized(vertices[static_cast<std::size_t>(vertex)])
        : Vec3f{};
}

template <typename Fn>
void for_each_path_neighbor(const BoundaryPath& path, std::size_t index, Fn&& visit)
{
    if (index > 0) {
        visit(path.ids[index - 1]);
    } else if (path.closed && path.ids.size() > 1) {
        visit(path.ids.back());
    }
    if (index + 1 < path.ids.size()) {
        visit(path.ids[index + 1]);
    } else if (path.closed && path.ids.size() > 1 && path.ids.front() != path.ids.back()) {
        visit(path.ids.front());
    }
}

Vec3f segment_open_dir(
    int vertex,
    Vec3f tangent,
    const std::vector<Vec3f>& vertices,
    const std::vector<Vec3f>& vertex_normals
)
{
    Vec3f open = normalized(cross(tangent, normal_for_vertex(vertex, vertices, vertex_normals)));
    if (squared_norm(open) <= 0.0) {
        open = normalized(cross(tangent, normalized(vertices[static_cast<std::size_t>(vertex)])));
    }
    return open;
}

}  // namespace

BoundaryIncidenceData build_boundary_incidence(
    const std::vector<BoundaryPath>& paths,
    const std::vector<Edge>& boundary_edges,
    const std::vector<Vec3f>& vertices,
    const std::vector<Vec3f>& vertex_normals
)
{
    BoundaryIncidenceData data;
    std::size_t expected_vertex_count = 0;
    for (const BoundaryPath& path : paths) {
        expected_vertex_count += path.ids.size();
    }
    data.by_vertex.reserve(expected_vertex_count);
    std::unordered_map<std::uint64_t, BoundaryEdgeGeometry> edge_lookup;
    edge_lookup.reserve(boundary_edges.size());
    for (const Edge edge : boundary_edges) {
        const auto [iterator, inserted] = edge_lookup.emplace(
            edge_key(edge[0], edge[1]), BoundaryEdgeGeometry{edge}
        );
        if (inserted) {
            iterator->second.tangent = normalized(subtract(
                vertices[static_cast<std::size_t>(edge[1])],
                vertices[static_cast<std::size_t>(edge[0])]
            ));
            iterator->second.angular_tangent = angular_edge_tangent(edge, vertices);
        }
    }
    std::unordered_map<std::uint64_t, std::pair<Vec3f, int>> segment_accum;
    segment_accum.reserve(boundary_edges.size());
    for (const BoundaryPath& path : paths) {
        for (std::size_t idx = 0; idx < path.ids.size(); ++idx) {
            const int vertex = path.ids[idx];
            for_each_path_neighbor(path, idx, [&](int neighbor) {
                const auto edge_iter = edge_lookup.find(edge_key(vertex, neighbor));
                if (edge_iter == edge_lookup.end()) {
                    return;
                }
                const BoundaryEdgeGeometry& geometry = edge_iter->second;
                const Vec3f open = segment_open_dir(
                    vertex, geometry.tangent, vertices, vertex_normals
                );
                const Vec2d angular_tangent = geometry.angular_tangent;
                auto [vertex_iter, inserted] = data.by_vertex.try_emplace(vertex);
                if (inserted) {
                    vertex_iter->second.reserve(2);
                }
                vertex_iter->second.push_back(BoundaryIncidence{
                    neighbor,
                    geometry.tangent,
                    open,
                    angular_tangent,
                    angular_exterior_direction(angular_tangent),
                    -1,
                });
                const std::uint64_t key = edge_key(vertex, neighbor);
                auto& accumulation = segment_accum[key];
                accumulation.first = add(accumulation.first, open);
                ++accumulation.second;
            });
        }
    }
    data.segment_open_dirs.reserve(segment_accum.size());
    for (const auto& [key, accumulation] : segment_accum) {
        const Vec3f average = scale(accumulation.first, 1.0 / std::max(1, accumulation.second));
        const Vec3f direction = normalized(average);
        if (squared_norm(direction) > 0.0) {
            data.segment_open_dirs.emplace(key, direction);
        }
    }
    return data;
}

void annotate_open_side_components(
    BoundaryIncidenceData& incidence,
    const StitchConnectionOracle& connection_oracle,
    bool source_is_old
)
{
    struct OpenSideWork {
        int vertex{};
        BoundaryIncidence* item{};
    };
    std::vector<OpenSideWork> work;
    for (auto& [vertex, items] : incidence.by_vertex) {
        for (BoundaryIncidence& item : items) {
            work.push_back(OpenSideWork{vertex, &item});
        }
    }
#if defined(RI_MESHING_HAS_OPENMP)
    #pragma omp parallel for if (work.size() > 64) schedule(static)
#endif
    for (int index = 0; index < static_cast<int>(work.size()); ++index) {
        const OpenSideWork& work_item = work[static_cast<std::size_t>(index)];
        BoundaryIncidence& item = *work_item.item;
        item.frontier_component = connection_oracle.component_for_open_side(
            work_item.vertex,
            item.angular_open_dir.x,
            item.angular_open_dir.y,
            source_is_old
        );
    }
    incidence.frontier_components_annotated = true;
}

void annotate_open_side_component_pair(
    BoundaryIncidenceData& old_incidence,
    BoundaryIncidenceData& new_incidence,
    const StitchConnectionOracle& connection_oracle
)
{
    struct OpenSideWork {
        int vertex{};
        BoundaryIncidence* item{};
        bool source_is_old{};
    };
    std::vector<OpenSideWork> work;
    for (auto& [vertex, items] : old_incidence.by_vertex) {
        for (BoundaryIncidence& item : items) {
            work.push_back(OpenSideWork{vertex, &item, true});
        }
    }
    for (auto& [vertex, items] : new_incidence.by_vertex) {
        for (BoundaryIncidence& item : items) {
            work.push_back(OpenSideWork{vertex, &item, false});
        }
    }
#if defined(RI_MESHING_HAS_OPENMP)
    #pragma omp parallel for if (work.size() > 64) schedule(static)
#endif
    for (int index = 0; index < static_cast<int>(work.size()); ++index) {
        const OpenSideWork& work_item = work[static_cast<std::size_t>(index)];
        work_item.item->frontier_component = connection_oracle.component_for_open_side(
            work_item.vertex,
            work_item.item->angular_open_dir.x,
            work_item.item->angular_open_dir.y,
            work_item.source_is_old
        );
    }
    old_incidence.frontier_components_annotated = true;
    new_incidence.frontier_components_annotated = true;
}

bool has_open_side_match(
    int old_vertex,
    int new_vertex,
    const std::vector<Vec3f>& vertices,
    const BoundaryIncidenceData& old_incidence,
    const BoundaryIncidenceData& new_incidence
)
{
    const auto old_iter = old_incidence.by_vertex.find(old_vertex);
    const auto new_iter = new_incidence.by_vertex.find(new_vertex);
    if (old_iter == old_incidence.by_vertex.end() || new_iter == new_incidence.by_vertex.end()) {
        return true;
    }
    if (old_incidence.frontier_components_annotated
        && new_incidence.frontier_components_annotated) {
        for (const BoundaryIncidence& old_item : old_iter->second) {
            if (old_item.frontier_component < 0) {
                continue;
            }
            for (const BoundaryIncidence& new_item : new_iter->second) {
                if (old_item.frontier_component == new_item.frontier_component) {
                    return true;
                }
            }
        }
        return false;
    }
    const Vec2d direction = angular_direction(
        vertices[static_cast<std::size_t>(old_vertex)],
        vertices[static_cast<std::size_t>(new_vertex)]
    );
    if (squared_norm(direction) <= 1e-24) {
        return false;
    }
    for (const BoundaryIncidence& old_incidence_item : old_iter->second) {
        for (const BoundaryIncidence& new_incidence_item : new_iter->second) {
            if (dot(
                    old_incidence_item.angular_tangent,
                    new_incidence_item.angular_tangent
                ) >= 0.0) {
                continue;
            }
            const MeshScalar side_old = dot(old_incidence_item.angular_open_dir, direction);
            const MeshScalar side_new = dot(
                new_incidence_item.angular_open_dir,
                Vec2d{-direction.x, -direction.y}
            );
            if (side_old > -0.10 && side_new > -0.10) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace ri_meshing::stitch_detail
