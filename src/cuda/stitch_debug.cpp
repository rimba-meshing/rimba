#include "stitch_debug.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ri_meshing::cuda_pipeline {
namespace {

std::string snapshot_stem(const StitchDebugDump& debug_dump, int frame_index)
{
    const std::string label = debug_dump.label.empty() ? "stitch" : debug_dump.label;
    std::ostringstream stem;
    stem << label << "_frame_" << std::setw(6) << std::setfill('0') << frame_index;
    return stem.str();
}

std::string hole_metrics_stem(const HoleFillMetricsDump& debug_dump, int frame_index)
{
    const std::string label = debug_dump.label.empty() ? "hole-fill" : debug_dump.label;
    std::ostringstream stem;
    stem << label << "_frame_" << std::setw(6) << std::setfill('0') << frame_index;
    return stem.str();
}

void write_vertices(const std::filesystem::path& path, const std::vector<Vec3f>& vertices)
{
    std::ofstream out(path);
    out << "id,x,y,z\n";
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        const Vec3f point = vertices[index];
        out << index << ',' << point.x << ',' << point.y << ',' << point.z << '\n';
    }
}

void write_faces(const std::filesystem::path& path, const std::vector<Triangle>& faces)
{
    std::ofstream out(path);
    out << "v0,v1,v2\n";
    for (const Triangle face : faces) {
        out << face.v0 << ',' << face.v1 << ',' << face.v2 << '\n';
    }
}

void write_edges(const std::filesystem::path& path, const std::vector<std::array<int, 2>>& edges)
{
    std::ofstream out(path);
    out << "v0,v1\n";
    for (const auto edge : edges) {
        out << edge[0] << ',' << edge[1] << '\n';
    }
}

void write_paths(const std::filesystem::path& path, const std::vector<StitchDebugPath>& paths)
{
    std::ofstream out(path);
    out << "path,closed,pos,vertex\n";
    for (std::size_t path_index = 0; path_index < paths.size(); ++path_index) {
        const StitchDebugPath& item = paths[path_index];
        for (std::size_t position = 0; position < item.ids.size(); ++position) {
            out << path_index << ',' << (item.closed ? 1 : 0) << ',' << position << ',' << item.ids[position] << '\n';
        }
    }
}

void write_labels(const std::filesystem::path& path, const std::vector<StitchDebugLabel>& labels)
{
    std::ofstream out(path);
    out << "vertex,cluster,label,path_t,score\n";
    for (const StitchDebugLabel& item : labels) {
        out << item.vertex_id << ',' << item.cluster << ',' << item.label << ','
            << item.path_t << ',' << item.score << '\n';
    }
}

void write_segments(const std::filesystem::path& path, const std::vector<StitchDebugSegment>& segments)
{
    std::ofstream out(path);
    out << "cluster,segment,pos,vertex,label\n";
    for (const StitchDebugSegment& item : segments) {
        for (std::size_t position = 0; position < item.ids.size(); ++position) {
            out << item.cluster << ',' << item.segment << ',' << position << ','
                << item.ids[position] << ',' << item.labels[position] << '\n';
        }
    }
}

int remap_id(int id, const std::vector<int>& compact_to_global)
{
    if (id < 0 || static_cast<std::size_t>(id) >= compact_to_global.size()) {
        return -1;
    }
    return compact_to_global[static_cast<std::size_t>(id)];
}

void remap_edges(std::vector<std::array<int, 2>>& edges, const std::vector<int>& compact_to_global)
{
    for (auto& edge : edges) {
        edge[0] = remap_id(edge[0], compact_to_global);
        edge[1] = remap_id(edge[1], compact_to_global);
    }
}

void remap_faces(std::vector<Triangle>& faces, const std::vector<int>& compact_to_global)
{
    for (Triangle& face : faces) {
        face.v0 = remap_id(face.v0, compact_to_global);
        face.v1 = remap_id(face.v1, compact_to_global);
        face.v2 = remap_id(face.v2, compact_to_global);
    }
}

void write_snapshot_csv(const std::filesystem::path& base, const StitchFrameDebugSnapshot& snapshot)
{
    write_vertices(base.string() + "_vertices.csv", snapshot.vertices_global);
    write_faces(base.string() + "_faces_before_stitch.csv", snapshot.faces_before_stitch);
    write_faces(base.string() + "_new_faces.csv", snapshot.new_faces);
    write_faces(base.string() + "_stitch_faces.csv", snapshot.stitch_faces);
    write_faces(base.string() + "_hole_faces.csv", snapshot.hole_faces);
    write_edges(base.string() + "_old_boundary_edges.csv", snapshot.core.old_boundary_edges);
    write_edges(base.string() + "_new_boundary_edges.csv", snapshot.core.new_boundary_edges);
    write_paths(base.string() + "_old_paths.csv", snapshot.core.old_paths);
    write_paths(base.string() + "_new_paths.csv", snapshot.core.new_paths);
    write_labels(base.string() + "_old_labels.csv", snapshot.core.old_labels);
    write_labels(base.string() + "_new_labels.csv", snapshot.core.new_labels);
    write_segments(base.string() + "_old_segments.csv", snapshot.core.old_segments);
    write_faces(base.string() + "_candidate_faces_before_filter.csv", snapshot.core.candidate_faces_before_filter);
    write_faces(base.string() + "_candidate_faces_after_filter.csv", snapshot.core.candidate_faces_after_filter);
    write_faces(base.string() + "_selected_faces_local.csv", snapshot.core.selected_faces_local);
}

void write_snapshot_metadata(const std::filesystem::path& base, const std::string& stem, const StitchFrameDebugSnapshot& snapshot)
{
    std::ofstream out(base.string() + "_meta.json");
    out << "{\n";
    out << "  \"frame\": " << snapshot.frame_index << ",\n";
    out << "  \"stem\": \"" << stem << "\",\n";
    out << "  \"vertices\": " << snapshot.vertices_global.size() << ",\n";
    out << "  \"faces_before_stitch\": " << snapshot.faces_before_stitch.size() << ",\n";
    out << "  \"new_faces\": " << snapshot.new_faces.size() << ",\n";
    out << "  \"stitch_faces\": " << snapshot.stitch_faces.size() << ",\n";
    out << "  \"hole_faces\": " << snapshot.hole_faces.size() << ",\n";
    out << "  \"old_boundary_edges\": " << snapshot.core.old_boundary_edges.size() << ",\n";
    out << "  \"new_boundary_edges\": " << snapshot.core.new_boundary_edges.size() << ",\n";
    out << "  \"old_labels\": " << snapshot.core.old_labels.size() << ",\n";
    out << "  \"new_labels\": " << snapshot.core.new_labels.size() << ",\n";
    out << "  \"old_segments\": " << snapshot.core.old_segments.size() << ",\n";
    out << "  \"candidate_faces_before_filter\": " << snapshot.core.candidate_faces_before_filter.size() << ",\n";
    out << "  \"candidate_faces_after_filter\": " << snapshot.core.candidate_faces_after_filter.size() << '\n';
    out << "}\n";
}

void update_manifest(const StitchDebugDump& debug_dump, const std::string& stem, const StitchFrameDebugSnapshot& snapshot)
{
    std::ofstream manifest(debug_dump.directory / "manifest.jsonl", std::ios::app);
    manifest << "{\"frame\":" << snapshot.frame_index
             << ",\"stem\":\"" << stem
             << "\",\"vertices\":" << snapshot.vertices_global.size()
             << ",\"faces_before_stitch\":" << snapshot.faces_before_stitch.size()
             << ",\"new_faces\":" << snapshot.new_faces.size()
             << ",\"stitch_faces\":" << snapshot.stitch_faces.size()
             << ",\"hole_faces\":" << snapshot.hole_faces.size()
             << "}\n";
    std::ofstream descriptor(debug_dump.directory / "manifest.json");
    descriptor << "{\n";
    descriptor << "  \"format\": \"ri_meshing_stitch_debug_csv_v1\",\n";
    descriptor << "  \"index\": \"manifest.jsonl\"\n";
    descriptor << "}\n";
}

void write_hole_metrics_csv(
    const std::filesystem::path& path,
    int frame_index,
    const HoleFillResult& result
)
{
    std::ofstream out(path);
    out << "frame,path_index,edge_count,perimeter_3d_m,ri_area_pixels,diameter_3d_m,"
           "closed,mixed_old_new,basic_valid,perimeter_valid,projection_valid,ri_area_valid,seam_supported,"
           "hole_winding,diameter_exact,center_attempted,center_accepted,fallback_attempted,fallback_accepted,"
           "accepted,generated_faces\n";
    out << std::setprecision(17);
    for (const HoleFillLoopMetrics& metrics : result.loop_metrics) {
        out << frame_index << ','
            << metrics.path_index << ','
            << metrics.edge_count << ','
            << metrics.perimeter_3d_m << ','
            << metrics.ri_area_pixels << ','
            << metrics.diameter_3d_m << ','
            << (metrics.closed ? 1 : 0) << ','
            << (metrics.mixed_old_new ? 1 : 0) << ','
            << (metrics.basic_valid ? 1 : 0) << ','
            << (metrics.perimeter_valid ? 1 : 0) << ','
            << (metrics.projection_valid ? 1 : 0) << ','
            << (metrics.ri_area_valid ? 1 : 0) << ','
            << (metrics.seam_supported ? 1 : 0) << ','
            << (metrics.hole_winding ? 1 : 0) << ','
            << (metrics.diameter_exact ? 1 : 0) << ','
            << (metrics.center_attempted ? 1 : 0) << ','
            << (metrics.center_accepted ? 1 : 0) << ','
            << (metrics.fallback_attempted ? 1 : 0) << ','
            << (metrics.fallback_accepted ? 1 : 0) << ','
            << (metrics.accepted ? 1 : 0) << ','
            << metrics.generated_faces << '\n';
    }
}

}  // namespace

void remap_stitch_core_debug(StitchCoreDebug& debug, const std::vector<int>& compact_to_global)
{
    remap_edges(debug.old_boundary_edges, compact_to_global);
    remap_edges(debug.new_boundary_edges, compact_to_global);
    for (StitchDebugPath& path : debug.old_paths) {
        for (int& id : path.ids) {
            id = remap_id(id, compact_to_global);
        }
    }
    for (StitchDebugPath& path : debug.new_paths) {
        for (int& id : path.ids) {
            id = remap_id(id, compact_to_global);
        }
    }
    for (StitchDebugLabel& label : debug.old_labels) {
        label.vertex_id = remap_id(label.vertex_id, compact_to_global);
    }
    for (StitchDebugLabel& label : debug.new_labels) {
        label.vertex_id = remap_id(label.vertex_id, compact_to_global);
    }
    for (StitchDebugSegment& segment : debug.old_segments) {
        for (int& id : segment.ids) {
            id = remap_id(id, compact_to_global);
        }
    }
    remap_faces(debug.candidate_faces_before_filter, compact_to_global);
    remap_faces(debug.candidate_faces_after_filter, compact_to_global);
    remap_faces(debug.selected_faces_local, compact_to_global);
}

void dump_stitch_debug_snapshot(const StitchDebugDump& debug_dump, const StitchFrameDebugSnapshot& snapshot)
{
    std::filesystem::create_directories(debug_dump.directory);
    const std::string stem = snapshot_stem(debug_dump, snapshot.frame_index);
    const std::filesystem::path base = debug_dump.directory / stem;
    write_snapshot_csv(base, snapshot);
    write_snapshot_metadata(base, stem, snapshot);
    update_manifest(debug_dump, stem, snapshot);
}

void dump_hole_fill_metrics(
    const HoleFillMetricsDump& debug_dump,
    int frame_index,
    const HoleFillResult& result
)
{
    std::filesystem::create_directories(debug_dump.directory);
    const std::string stem = hole_metrics_stem(debug_dump, frame_index);
    write_hole_metrics_csv(debug_dump.directory / (stem + "_loops.csv"), frame_index, result);
}

}  // namespace ri_meshing::cuda_pipeline
