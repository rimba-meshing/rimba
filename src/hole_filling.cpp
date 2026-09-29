#include "ri_meshing/hole_filling.hpp"

#include "ri_meshing/face_orientation.hpp"
#include "ri_meshing/geometry.hpp"
#include "ri_meshing/timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace ri_meshing {
namespace {

struct Point2d {
    MeshScalar x{};
    MeshScalar y{};
};

struct ProjectedLoop {
    std::vector<int> ids;
    std::vector<Point2d> points;
    Point2d center{};
    MeshScalar signed_area{};
};

Vec3f transform_point(Vec3f point, const Mat4f& transform)
{
    const MeshScalar x = transform[0] * point.x + transform[1] * point.y + transform[2] * point.z + transform[3];
    const MeshScalar y = transform[4] * point.x + transform[5] * point.y + transform[6] * point.z + transform[7];
    const MeshScalar z = transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11];
    const MeshScalar w = transform[12] * point.x + transform[13] * point.y + transform[14] * point.z + transform[15];
    const MeshScalar inverse_w = std::abs(w) > static_cast<MeshScalar>(1e-12)
        ? static_cast<MeshScalar>(1) / w : static_cast<MeshScalar>(1);
    return Vec3f{
        static_cast<float>(x * inverse_w),
        static_cast<float>(y * inverse_w),
        static_cast<float>(z * inverse_w),
    };
}

constexpr MeshScalar kEpsilon = static_cast<MeshScalar>(1e-10);

MeshScalar cross(Point2d a, Point2d b, Point2d c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool point_on_segment(Point2d point, Point2d a, Point2d b)
{
    if (std::abs(cross(a, b, point)) > 1e-8) {
        return false;
    }
    return point.x >= std::min(a.x, b.x) - kEpsilon
        && point.x <= std::max(a.x, b.x) + kEpsilon
        && point.y >= std::min(a.y, b.y) - kEpsilon
        && point.y <= std::max(a.y, b.y) + kEpsilon;
}

bool point_in_polygon(Point2d point, const std::vector<Point2d>& polygon)
{
    bool inside = false;
    for (std::size_t index = 0; index < polygon.size(); ++index) {
        const Point2d a = polygon[index];
        const Point2d b = polygon[(index + 1) % polygon.size()];
        if (point_on_segment(point, a, b)) {
            return true;
        }
        const bool crosses = (a.y > point.y) != (b.y > point.y);
        if (crosses) {
            const MeshScalar x = a.x + (b.x - a.x) * (point.y - a.y) / (b.y - a.y);
            if (point.x < x) {
                inside = !inside;
            }
        }
    }
    return inside;
}

bool segments_intersect(Point2d a, Point2d b, Point2d c, Point2d d)
{
    const MeshScalar ab_c = cross(a, b, c);
    const MeshScalar ab_d = cross(a, b, d);
    const MeshScalar cd_a = cross(c, d, a);
    const MeshScalar cd_b = cross(c, d, b);
    const bool proper = ((ab_c > kEpsilon && ab_d < -kEpsilon)
        || (ab_c < -kEpsilon && ab_d > kEpsilon))
        && ((cd_a > kEpsilon && cd_b < -kEpsilon)
            || (cd_a < -kEpsilon && cd_b > kEpsilon));
    return proper || point_on_segment(c, a, b) || point_on_segment(d, a, b)
        || point_on_segment(a, c, d) || point_on_segment(b, c, d);
}

bool simple_polygon(const std::vector<Point2d>& points)
{
    for (std::size_t first = 0; first < points.size(); ++first) {
        const std::size_t first_next = (first + 1) % points.size();
        for (std::size_t second = first + 1; second < points.size(); ++second) {
            const std::size_t second_next = (second + 1) % points.size();
            if (first == second || first_next == second || second_next == first) {
                continue;
            }
            if (segments_intersect(
                    points[first], points[first_next], points[second], points[second_next])) {
                return false;
            }
        }
    }
    return true;
}

bool project_local_point(Vec3f point, const SparseMeshingConfig& config, Point2d& projected)
{
    const MeshScalar pi = static_cast<MeshScalar>(std::acos(-1.0));
    const MeshScalar xy = std::hypot(point.x, point.y);
    const MeshScalar range = std::sqrt(
        static_cast<MeshScalar>(point.x) * point.x
        + static_cast<MeshScalar>(point.y) * point.y
        + static_cast<MeshScalar>(point.z) * point.z
    );
    if (range <= static_cast<MeshScalar>(0)) {
        return false;
    }
    const MeshScalar elevation = std::atan2(static_cast<MeshScalar>(point.z), xy);
    const MeshScalar azimuth = std::atan2(static_cast<MeshScalar>(point.y), static_cast<MeshScalar>(point.x));
    const MeshScalar fov_range = config.vertical_fov.max_rad - config.vertical_fov.min_rad;
    const int row = static_cast<int>(std::llround(
        (elevation - config.vertical_fov.min_rad) / fov_range
            * static_cast<MeshScalar>(config.ri_height)
    ));
    if (row < 0 || row >= static_cast<int>(config.ri_height)) {
        return false;
    }
    const int col = static_cast<int>(std::llround(
        (azimuth + pi) / (static_cast<MeshScalar>(2) * pi)
            * static_cast<MeshScalar>(config.ri_width)
    ));
    projected = Point2d{static_cast<MeshScalar>(col), static_cast<MeshScalar>(row)};
    return true;
}

MeshScalar polygon_signed_area(const std::vector<Point2d>& points)
{
    MeshScalar area = static_cast<MeshScalar>(0);
    for (std::size_t index = 0; index < points.size(); ++index) {
        const Point2d a = points[index];
        const Point2d b = points[(index + 1) % points.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return static_cast<MeshScalar>(0.5) * area;
}

ProjectedLoop project_loop(
    const BoundaryPath& path,
    const std::vector<Vec3f>& vertices,
    const Mat4f& global_to_local,
    const SparseMeshingConfig& config
)
{
    ProjectedLoop result;
    result.ids.reserve(path.ids.size());
    result.points.reserve(path.ids.size());
    const int width = static_cast<int>(config.ri_width);
    Vec3f center_local{};
    for (const int id : path.ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= vertices.size()) {
            return {};
        }
        const Vec3f point = transform_point(vertices[static_cast<std::size_t>(id)], global_to_local);
        center_local.x += point.x;
        center_local.y += point.y;
        center_local.z += point.z;
        Point2d projected;
        if (!project_local_point(point, config, projected)) {
            return {};
        }
        result.ids.push_back(id);
        result.points.push_back(projected);
    }
    if (result.points.size() < 3) {
        return {};
    }
    const MeshScalar inverse_size = static_cast<MeshScalar>(1) / result.points.size();
    center_local.x = static_cast<float>(center_local.x * inverse_size);
    center_local.y = static_cast<float>(center_local.y * inverse_size);
    center_local.z = static_cast<float>(center_local.z * inverse_size);
    if (!project_local_point(center_local, config, result.center)) {
        return {};
    }
    for (std::size_t index = 1; index < result.points.size(); ++index) {
        MeshScalar delta = result.points[index].x - result.points[index - 1].x;
        if (delta > static_cast<MeshScalar>(width) / static_cast<MeshScalar>(2)) {
            result.points[index].x -= static_cast<MeshScalar>(width);
        } else if (delta < -static_cast<MeshScalar>(width) / static_cast<MeshScalar>(2)) {
            result.points[index].x += static_cast<MeshScalar>(width);
        }
    }
    while (result.center.x - result.points.front().x
           > static_cast<MeshScalar>(width) / static_cast<MeshScalar>(2)) {
        result.center.x -= static_cast<MeshScalar>(width);
    }
    while (result.center.x - result.points.front().x
           < -static_cast<MeshScalar>(width) / static_cast<MeshScalar>(2)) {
        result.center.x += static_cast<MeshScalar>(width);
    }
    result.signed_area = polygon_signed_area(result.points);
    return result;
}

bool seam_pixel_at(const StitchConnectionMask& mask, int row, int col)
{
    if (row < 0 || row >= mask.height || mask.width <= 0) {
        return false;
    }
    col %= mask.width;
    if (col < 0) {
        col += mask.width;
    }
    return mask.pixels[static_cast<std::size_t>(row * mask.width + col)] != 0;
}

bool has_seam_support(const ProjectedLoop& loop, const StitchConnectionMask& mask)
{
    if (mask.empty() || loop.points.empty()) {
        return false;
    }
    MeshScalar min_x = loop.points.front().x;
    MeshScalar max_x = min_x;
    MeshScalar min_y = loop.points.front().y;
    MeshScalar max_y = min_y;
    for (const Point2d point : loop.points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }
    if (max_x - min_x > static_cast<MeshScalar>(mask.width) / static_cast<MeshScalar>(2)
        || min_y < static_cast<MeshScalar>(0)
        || max_y >= static_cast<MeshScalar>(mask.height)) {
        return false;
    }
    const int first_row = std::max(0, static_cast<int>(std::floor(min_y)));
    const int last_row = std::min(mask.height - 1, static_cast<int>(std::ceil(max_y)));
    const int first_col = static_cast<int>(std::floor(min_x));
    const int last_col = static_cast<int>(std::ceil(max_x));
    for (int row = first_row; row <= last_row; ++row) {
        for (int col = first_col; col <= last_col; ++col) {
            if (!point_in_polygon(Point2d{
                    static_cast<MeshScalar>(col) + static_cast<MeshScalar>(0.5),
                    static_cast<MeshScalar>(row) + static_cast<MeshScalar>(0.5)}, loop.points)) {
                continue;
            }
            for (int row_offset = -1; row_offset <= 1; ++row_offset) {
                for (int col_offset = -1; col_offset <= 1; ++col_offset) {
                    if (seam_pixel_at(mask, row + row_offset, col + col_offset)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

std::vector<Triangle> triangulate(ProjectedLoop loop)
{
    const MeshScalar area = static_cast<MeshScalar>(2) * loop.signed_area;
    if (std::abs(area) <= kEpsilon) {
        return {};
    }
    if (area < static_cast<MeshScalar>(0)) {
        std::reverse(loop.ids.begin(), loop.ids.end());
        std::reverse(loop.points.begin(), loop.points.end());
    }
    std::vector<std::size_t> remaining(loop.ids.size());
    for (std::size_t index = 0; index < remaining.size(); ++index) {
        remaining[index] = index;
    }
    std::vector<Triangle> faces;
    faces.reserve(loop.ids.size() - 2);
    while (remaining.size() > 3) {
        bool found_ear = false;
        for (std::size_t position = 0; position < remaining.size(); ++position) {
            const std::size_t previous = remaining[(position + remaining.size() - 1) % remaining.size()];
            const std::size_t current = remaining[position];
            const std::size_t next = remaining[(position + 1) % remaining.size()];
            if (cross(loop.points[previous], loop.points[current], loop.points[next]) <= kEpsilon) {
                continue;
            }
            bool contains_point = false;
            for (const std::size_t candidate : remaining) {
                if (candidate == previous || candidate == current || candidate == next) {
                    continue;
                }
                if (point_in_polygon(loop.points[candidate], {
                        loop.points[previous], loop.points[current], loop.points[next]})) {
                    contains_point = true;
                    break;
                }
            }
            if (contains_point) {
                continue;
            }
            faces.push_back(Triangle{
                loop.ids[previous], loop.ids[current], loop.ids[next]
            });
            remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(position));
            found_ear = true;
            break;
        }
        if (!found_ear) {
            return {};
        }
    }
    faces.push_back(Triangle{
        loop.ids[remaining[0]], loop.ids[remaining[1]], loop.ids[remaining[2]]
    });
    return faces;
}

std::vector<Triangle> triangulate_with_center(const ProjectedLoop& loop, int center_id)
{
    if (!point_in_polygon(loop.center, loop.points)) {
        return {};
    }
    std::vector<Triangle> faces;
    faces.reserve(loop.ids.size());
    for (std::size_t index = 0; index < loop.ids.size(); ++index) {
        faces.push_back(Triangle{
            loop.ids[index],
            loop.ids[(index + 1) % loop.ids.size()],
            center_id,
        });
    }
    return faces;
}

Vec3f loop_center_global(const BoundaryPath& path, const std::vector<Vec3f>& vertices)
{
    Vec3f center{};
    for (const int id : path.ids) {
        const Vec3f point = vertices[static_cast<std::size_t>(id)];
        center.x += point.x;
        center.y += point.y;
        center.z += point.z;
    }
    const float inverse_size = 1.0F / static_cast<float>(path.ids.size());
    return Vec3f{center.x * inverse_size, center.y * inverse_size, center.z * inverse_size};
}

Vec3f point_for_id(
    int id,
    const HoleFillRequest& request,
    const std::vector<Vec3f>& extra_vertices
)
{
    if (id < 0) {
        return {};
    }
    const std::size_t base_size = request.vertices_global.size();
    if (static_cast<std::size_t>(id) < base_size) {
        return request.vertices_global[static_cast<std::size_t>(id)];
    }
    const std::size_t extra_id = static_cast<std::size_t>(id) - base_size;
    return extra_id < extra_vertices.size() ? extra_vertices[extra_id] : Vec3f{};
}

bool validate_patch(
    std::vector<Triangle>& patch,
    const HoleFillRequest& request,
    const std::vector<Vec3f>& extra_vertices
)
{
    for (Triangle& face : patch) {
        const Vec3f a = transform_point(point_for_id(face.v0, request, extra_vertices), request.global_to_local);
        const Vec3f b = transform_point(point_for_id(face.v1, request, extra_vertices), request.global_to_local);
        const Vec3f c = transform_point(point_for_id(face.v2, request, extra_vertices), request.global_to_local);
        const TriangleMetrics metrics = triangle_metrics(a, b, c);
        if (metrics.longest_edge_sq
                > request.config.hole_fill_max_triangle_dist
                      * request.config.hole_fill_max_triangle_dist
            || 0.5 * metrics.twice_area < request.config.sliver_repair_min_area
            || metrics.quality < request.config.sliver_repair_min_quality) {
            return false;
        }
        if (sensor_orientation_dot(a, b, c) > 0.0) {
            std::swap(face.v1, face.v2);
        }
    }
    return true;
}

bool loop_is_mixed(const BoundaryPath& path, int old_vertex_count)
{
    bool has_old = false;
    bool has_new = false;
    for (const int id : path.ids) {
        has_old = has_old || id < old_vertex_count;
        has_new = has_new || id >= old_vertex_count;
    }
    return has_old && has_new;
}

MeshScalar loop_perimeter(
    const BoundaryPath& path,
    const std::vector<Vec3f>& vertices,
    const Mat4f& global_to_local
)
{
    MeshScalar perimeter = static_cast<MeshScalar>(0);
    for (std::size_t index = 0; index < path.ids.size(); ++index) {
        const Vec3f a = transform_point(vertices[static_cast<std::size_t>(path.ids[index])], global_to_local);
        const Vec3f b = transform_point(
            vertices[static_cast<std::size_t>(path.ids[(index + 1) % path.ids.size()])], global_to_local
        );
        perimeter += std::sqrt(
            static_cast<MeshScalar>(a.x - b.x) * (a.x - b.x)
            + static_cast<MeshScalar>(a.y - b.y) * (a.y - b.y)
            + static_cast<MeshScalar>(a.z - b.z) * (a.z - b.z)
        );
    }
    return perimeter;
}

bool valid_loop_ids(const BoundaryPath& path, const std::vector<Vec3f>& vertices)
{
    return std::all_of(path.ids.begin(), path.ids.end(), [&](const int id) {
        return id >= 0 && static_cast<std::size_t>(id) < vertices.size();
    });
}

std::vector<Vec3f> transform_loop_points(
    const BoundaryPath& path,
    const std::vector<Vec3f>& vertices,
    const Mat4f& global_to_local
)
{
    std::vector<Vec3f> points;
    if (!valid_loop_ids(path, vertices)) {
        return points;
    }
    points.reserve(path.ids.size());
    for (const int id : path.ids) {
        points.push_back(transform_point(vertices[static_cast<std::size_t>(id)], global_to_local));
    }
    return points;
}

MeshScalar loop_diameter(const std::vector<Vec3f>& points)
{
    MeshScalar diameter_sq = static_cast<MeshScalar>(0);
    for (std::size_t first = 0; first < points.size(); ++first) {
        for (std::size_t second = first + 1; second < points.size(); ++second) {
            const MeshScalar dx = points[first].x - points[second].x;
            const MeshScalar dy = points[first].y - points[second].y;
            const MeshScalar dz = points[first].z - points[second].z;
            diameter_sq = std::max(diameter_sq, dx * dx + dy * dy + dz * dz);
        }
    }
    return std::sqrt(diameter_sq);
}

}  // namespace

HoleFillResult fill_local_seam_holes(const HoleFillRequest& request)
{
    HoleFillResult result;
    if (request.seam_mask == nullptr || request.seam_mask->empty()) {
        return result;
    }
    double geometry_milliseconds = 0.0;
    double support_milliseconds = 0.0;
    double triangulation_milliseconds = 0.0;
    const Stopwatch total_stopwatch;
    const bool instrument = request.timing != nullptr;
    if (request.collect_loop_metrics) {
        result.loop_metrics.reserve(request.boundary_paths.size());
    }
    for (std::size_t path_index = 0; path_index < request.boundary_paths.size(); ++path_index) {
        const BoundaryPath& path = request.boundary_paths[path_index];
        HoleFillLoopMetrics metrics;
        metrics.path_index = path_index;
        metrics.edge_count = path.ids.size();
        metrics.closed = path.closed;
        metrics.mixed_old_new = loop_is_mixed(path, request.old_vertex_count);
        const auto finalize_metrics = [&] {
            if (request.collect_loop_metrics) {
                result.loop_metrics.push_back(metrics);
            }
        };
        if (request.collect_loop_metrics && valid_loop_ids(path, request.vertices_global)) {
            metrics.perimeter_3d_m = loop_perimeter(
                path, request.vertices_global, request.global_to_local
            );
            // Oversized paths are rejected by max_hole_edges and are not useful
            // for a quadratic exact-diameter diagnostic.
            if (path.ids.size() <= 512) {
                const std::vector<Vec3f> local_points = transform_loop_points(
                    path, request.vertices_global, request.global_to_local
                );
                metrics.diameter_3d_m = loop_diameter(local_points);
                metrics.diameter_exact = true;
            }
        }
        Stopwatch phase_stopwatch;
        const bool basic_valid = path.closed
            && path.ids.size() >= 3
            && path.ids.size() <= static_cast<std::size_t>(request.config.max_hole_edges)
            && loop_is_mixed(path, request.old_vertex_count);
        metrics.basic_valid = basic_valid;
        if (!basic_valid) {
            if (instrument) {
                geometry_milliseconds += phase_stopwatch.elapsed_milliseconds();
            }
            finalize_metrics();
            continue;
        }
        const MeshScalar perimeter = loop_perimeter(path, request.vertices_global, request.global_to_local);
        metrics.perimeter_3d_m = perimeter;
        metrics.perimeter_valid = perimeter <= request.config.max_hole_euclidean;
        if (!metrics.perimeter_valid) {
            if (instrument) {
                geometry_milliseconds += phase_stopwatch.elapsed_milliseconds();
            }
            finalize_metrics();
            continue;
        }
        const ProjectedLoop projected = project_loop(
            path, request.vertices_global, request.global_to_local, request.config
        );
        if (instrument) {
            geometry_milliseconds += phase_stopwatch.elapsed_milliseconds();
        }
        metrics.projection_valid = projected.ids.size() >= 3;
        if (metrics.projection_valid) {
            metrics.ri_area_pixels = std::abs(projected.signed_area);
        }
        if (projected.ids.size() < 3) {
            finalize_metrics();
            continue;
        }
        metrics.ri_area_valid = metrics.ri_area_pixels <= request.config.max_hole_ri_area_pixels;
        if (!metrics.ri_area_valid) {
            finalize_metrics();
            continue;
        }
        phase_stopwatch = Stopwatch{};
        const bool has_support = has_seam_support(projected, *request.seam_mask);
        if (instrument) {
            support_milliseconds += phase_stopwatch.elapsed_milliseconds();
        }
        metrics.seam_supported = has_support;
        if (!has_support) {
            finalize_metrics();
            continue;
        }
        ++result.support_loops;
        phase_stopwatch = Stopwatch{};
        // Sensor-facing faces project clockwise in (azimuth, elevation), so
        // a counter-clockwise boundary is the contour of an empty hole.
        if (projected.signed_area <= kEpsilon) {
            metrics.hole_winding = false;
            if (instrument) {
                triangulation_milliseconds += phase_stopwatch.elapsed_milliseconds();
            }
            finalize_metrics();
            continue;
        }
        metrics.hole_winding = true;
        ++result.candidate_loops;
        const Vec3f center_global = loop_center_global(path, request.vertices_global);
        const int center_id = static_cast<int>(
            request.vertices_global.size() + result.vertices_global.size()
        );
        ++result.center_attempts;
        metrics.center_attempted = true;
        std::vector<Triangle> patch = triangulate_with_center(projected, center_id);
        bool uses_center = !patch.empty();
        if (uses_center) {
            std::vector<Vec3f> candidate_vertices = result.vertices_global;
            candidate_vertices.push_back(center_global);
            if (!validate_patch(patch, request, candidate_vertices)) {
                patch.clear();
                uses_center = false;
            }
        }
        if (patch.empty() && simple_polygon(projected.points)) {
            ++result.fallback_attempts;
            metrics.fallback_attempted = true;
            patch = triangulate(projected);
            if (!patch.empty() && !validate_patch(patch, request, {})) {
                patch.clear();
            }
        }
        if (patch.empty()) {
            if (instrument) {
                triangulation_milliseconds += phase_stopwatch.elapsed_milliseconds();
            }
            finalize_metrics();
            continue;
        }
        if (uses_center) {
            result.vertices_global.push_back(center_global);
            ++result.center_accepted;
            metrics.center_accepted = true;
        } else {
            ++result.fallback_accepted;
            metrics.fallback_accepted = true;
        }
        result.faces.insert(result.faces.end(), patch.begin(), patch.end());
        ++result.accepted_loops;
        metrics.accepted = true;
        metrics.generated_faces = patch.size();
        if (instrument) {
            triangulation_milliseconds += phase_stopwatch.elapsed_milliseconds();
        }
        finalize_metrics();
    }
    if (instrument) {
        request.timing->add_stage("stitch.hole_fill.total_host", total_stopwatch.elapsed_milliseconds());
        request.timing->add_stage("stitch.hole_fill.geometry", geometry_milliseconds);
        request.timing->add_stage("stitch.hole_fill.triangulation", triangulation_milliseconds);
        request.timing->add_stage("stitch.hole_fill.support", support_milliseconds);
    }
    return result;
}

}  // namespace ri_meshing
