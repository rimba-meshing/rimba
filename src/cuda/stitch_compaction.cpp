#include "stitch_compaction.hpp"

#include "stitch_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ri_meshing::cuda_pipeline {

namespace {

std::uint64_t host_edge_key(int a, int b)
{
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

void add_stage(PipelineTiming* timing, const char* name, const Stopwatch& stopwatch)
{
    if (timing != nullptr) {
        timing->add_stage(name, stopwatch.elapsed_milliseconds());
    }
}

std::array<int, 2> remap_edge(
    std::array<int, 2> edge,
    const std::vector<int>& global_to_compact
)
{
    return {
        global_to_compact[static_cast<std::size_t>(edge[0])],
        global_to_compact[static_cast<std::size_t>(edge[1])],
    };
}

void build_compact_map(
    const StitchCompactionRequest& request,
    std::vector<int>& global_to_compact,
    CompactStitchInput& out,
    PipelineTiming* timing
)
{
    const Stopwatch stopwatch;
    Stopwatch sub_stopwatch;
    out.compact_to_global.clear();
    out.old_vertex_count = static_cast<int>(request.old_boundary.vertex_ids.size());
    out.compact_to_global.reserve(
        request.old_boundary.vertex_ids.size() + request.new_boundary.vertex_ids.size()
    );
    out.compact_to_global.insert(
        out.compact_to_global.end(), request.old_boundary.vertex_ids.begin(), request.old_boundary.vertex_ids.end()
    );
    out.compact_to_global.insert(
        out.compact_to_global.end(), request.new_boundary.vertex_ids.begin(), request.new_boundary.vertex_ids.end()
    );
    add_stage(timing, "stitch.compact.build_map.compose_ids", sub_stopwatch);
    sub_stopwatch = Stopwatch{};
    if (global_to_compact.size() < request.vertices_global.size()) {
        global_to_compact.resize(request.vertices_global.size(), -1);
    }
    // Every edge remapped below references a vertex in compact_to_global, so
    // each queried entry is overwritten in this loop. Keeping stale entries
    // for vertices not touched by this frame avoids scanning the full
    // accumulated mesh on every compact-input build.
    add_stage(timing, "stitch.compact.build_map.resize", sub_stopwatch);
    sub_stopwatch = Stopwatch{};
    for (std::size_t compact_id = 0; compact_id < out.compact_to_global.size(); ++compact_id) {
        global_to_compact[static_cast<std::size_t>(out.compact_to_global[compact_id])] = static_cast<int>(compact_id);
    }
    add_stage(timing, "stitch.compact.build_map.fill", sub_stopwatch);
    add_stage(timing, "stitch.compact.build_map", stopwatch);
}

void transform_compact_vertices(
    const StitchCompactionRequest& request,
    CompactStitchInput& out,
    PipelineTiming* timing
)
{
    const Stopwatch stopwatch;
    Stopwatch sub_stopwatch;
    out.vertices_local.clear();
    out.vertices_local.reserve(out.compact_to_global.size());
    add_stage(timing, "stitch.compact.transform_vertices.reserve", sub_stopwatch);
    sub_stopwatch = Stopwatch{};
    for (const int global_id : out.compact_to_global) {
        if (global_id < 0 || static_cast<std::size_t>(global_id) >= request.vertices_global.size()) {
            throw std::runtime_error("stitching referenced a vertex outside the accumulated host mirror");
        }
        out.vertices_local.push_back(transform_point_cpu(
            request.vertices_global[static_cast<std::size_t>(global_id)], request.global_to_local
        ));
    }
    add_stage(timing, "stitch.compact.transform_vertices.loop", sub_stopwatch);
    add_stage(timing, "stitch.compact.transform_vertices", stopwatch);
}

Vec3f transform_normal_cpu(Vec3f normal, const Mat4f& transform)
{
    return Vec3f{
        static_cast<float>(transform[0] * normal.x + transform[1] * normal.y + transform[2] * normal.z),
        static_cast<float>(transform[4] * normal.x + transform[5] * normal.y + transform[6] * normal.z),
        static_cast<float>(transform[8] * normal.x + transform[9] * normal.y + transform[10] * normal.z),
    };
}

MeshScalar squared_norm(Vec3f value)
{
    return static_cast<MeshScalar>(value.x) * value.x
        + static_cast<MeshScalar>(value.y) * value.y
        + static_cast<MeshScalar>(value.z) * value.z;
}

void build_compact_vertex_normals(
    const StitchCompactionRequest& request,
    CompactStitchInput& out,
    PipelineTiming* timing
)
{
    const Stopwatch stopwatch;
    out.vertex_normals_local.assign(out.vertices_local.size(), Vec3f{});
    for (std::size_t compact_id = 0; compact_id < out.compact_to_global.size(); ++compact_id) {
        const int global_id = out.compact_to_global[compact_id];
        if (global_id < 0 || static_cast<std::size_t>(global_id) >= request.normal_sums_global.size()) {
            continue;
        }
        out.vertex_normals_local[compact_id] = transform_normal_cpu(
            request.normal_sums_global[static_cast<std::size_t>(global_id)],
            request.global_to_local
        );
    }
    for (std::size_t idx = 0; idx < out.vertex_normals_local.size(); ++idx) {
        Vec3f& normal = out.vertex_normals_local[idx];
        const MeshScalar length_sq = squared_norm(normal);
        if (length_sq > static_cast<MeshScalar>(1e-24)) {
            const float inverse_length = static_cast<float>(static_cast<MeshScalar>(1) / std::sqrt(length_sq));
            normal.x *= inverse_length;
            normal.y *= inverse_length;
            normal.z *= inverse_length;
            continue;
        }
        const Vec3f position = out.vertices_local[idx];
        const MeshScalar position_sq = squared_norm(position);
        if (position_sq > static_cast<MeshScalar>(1e-24)) {
            const float inverse_length = static_cast<float>(static_cast<MeshScalar>(1) / std::sqrt(position_sq));
            normal = Vec3f{
                position.x * inverse_length,
                position.y * inverse_length,
                position.z * inverse_length,
            };
        }
    }
    add_stage(timing, "stitch.compact.vertex_normals", stopwatch);
}

void remap_compact_edges(
    const StitchCompactionRequest& request,
    const std::vector<int>& global_to_compact,
    CompactStitchInput& out,
    PipelineTiming* timing
)
{
    const Stopwatch stopwatch;
    out.old_boundary_edges.clear();
    out.new_boundary_edges.clear();
    Stopwatch sub_stopwatch;
    out.old_boundary_edges.reserve(request.old_boundary.edges.size());
    for (const std::array<int, 2> edge : request.old_boundary.edges) {
        out.old_boundary_edges.push_back(remap_edge(edge, global_to_compact));
    }
    add_stage(timing, "stitch.compact.remap_edges.old", sub_stopwatch);
    sub_stopwatch = Stopwatch{};
    out.new_boundary_edges.reserve(request.new_boundary.edges.size());
    for (const std::array<int, 2> edge : request.new_boundary.edges) {
        out.new_boundary_edges.push_back(remap_edge(edge, global_to_compact));
    }
    add_stage(timing, "stitch.compact.remap_edges.new", sub_stopwatch);
    add_stage(timing, "stitch.compact.remap_edges", stopwatch);
}

}  // namespace

void compact_stitch_input(
    const StitchCompactionRequest& request,
    CompactStitchInput& out,
    std::vector<int>& global_to_compact,
    PipelineTiming* timing
)
{
    const Stopwatch collect_stopwatch;
    (void)request.old_global_vertex_count;
    add_stage(timing, "stitch.compact.collect_ids", collect_stopwatch);
    build_compact_map(request, global_to_compact, out, timing);
    transform_compact_vertices(request, out, timing);
    build_compact_vertex_normals(request, out, timing);
    remap_compact_edges(request, global_to_compact, out, timing);
}

void remap_faces_to_global(std::vector<Triangle>& faces, const std::vector<int>& compact_to_global)
{
    for (Triangle& face : faces) {
        face = Triangle{
            compact_to_global.at(static_cast<std::size_t>(face.v0)),
            compact_to_global.at(static_cast<std::size_t>(face.v1)),
            compact_to_global.at(static_cast<std::size_t>(face.v2)),
        };
    }
}

std::vector<std::array<int, 2>> batch_boundary_edges_sorted(const std::vector<Triangle>& faces)
{
    struct KeyedEdge {
        std::uint64_t key{};
        std::array<int, 2> edge{};
    };

    std::vector<KeyedEdge> keyed_edges;
    keyed_edges.reserve(faces.size() * 3);
    for (const Triangle face : faces) {
        const std::array<int, 2> edge0{face.v0, face.v1};
        const std::array<int, 2> edge1{face.v1, face.v2};
        const std::array<int, 2> edge2{face.v2, face.v0};
        keyed_edges.push_back(KeyedEdge{host_edge_key(edge0[0], edge0[1]), edge0});
        keyed_edges.push_back(KeyedEdge{host_edge_key(edge1[0], edge1[1]), edge1});
        keyed_edges.push_back(KeyedEdge{host_edge_key(edge2[0], edge2[1]), edge2});
    }

    std::sort(keyed_edges.begin(), keyed_edges.end(), [](const KeyedEdge& a, const KeyedEdge& b) {
        return a.key < b.key;
    });

    std::vector<std::array<int, 2>> out;
    out.reserve(keyed_edges.size());
    for (std::size_t begin = 0; begin < keyed_edges.size();) {
        std::size_t end = begin + 1;
        while (end < keyed_edges.size() && keyed_edges[end].key == keyed_edges[begin].key) {
            ++end;
        }
        if (end == begin + 1) {
            out.push_back(keyed_edges[begin].edge);
        }
        begin = end;
    }
    return out;
}

}  // namespace ri_meshing::cuda_pipeline
