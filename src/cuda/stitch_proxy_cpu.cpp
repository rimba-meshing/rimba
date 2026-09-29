#include "stitch_proxy_cpu.hpp"

#include "stitch_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

namespace ri_meshing::cuda_pipeline {

struct ProxyRasterShape {
    int height{};
    int width{};
    int radius{};
};

struct RiPixel {
    int row{};
    int col{};
};

struct ProxyRasterTarget {
    std::vector<std::uint8_t>& mask;
    std::vector<float>& depth;
    ProxyRasterShape shape;
};

struct TriangleMaskRequest {
    ProjectedRiPoint p0;
    ProjectedRiPoint p1;
    ProjectedRiPoint p2;
    int height{};
    int width{};
};

ProjectedRiPoint project_global_to_ri(
    Vec3f point_global,
    const Mat4f& global_to_local,
    const SparseMeshingConfig& config
)
{
    constexpr MeshScalar kPi = static_cast<MeshScalar>(3.14159265358979323846264338327950288);
    const Vec3f point = transform_point_cpu(point_global, global_to_local);
    const MeshScalar x = point.x;
    const MeshScalar y = point.y;
    const MeshScalar z = point.z;
    const MeshScalar range = std::sqrt(x * x + y * y + z * z);
    if (!std::isfinite(range) || range <= static_cast<MeshScalar>(0)) {
        return {};
    }
    const int height = static_cast<int>(config.ri_height);
    const int width = static_cast<int>(config.ri_width);
    const MeshScalar xy = std::sqrt(x * x + y * y);
    const MeshScalar phi = std::atan2(z, xy);
    const MeshScalar theta = std::atan2(y, x);
    const MeshScalar fov_range = config.vertical_fov.max_rad - config.vertical_fov.min_rad;
    const int row = static_cast<int>(std::llround(
        (phi - config.vertical_fov.min_rad) / fov_range * static_cast<MeshScalar>(height)
    ));
    int col = static_cast<int>(std::llround(
        (theta + kPi) / (static_cast<MeshScalar>(2) * kPi) * static_cast<MeshScalar>(width)
    ));
    col = std::clamp(col, 0, width - 1);
    if (row < 0 || row >= height) {
        return {};
    }
    return ProjectedRiPoint{row, col, static_cast<float>(range), true};
}

int unwrap_col_near(int reference_col, int col, int width)
{
    int out = col;
    const int half_width = width / 2;
    if (out - reference_col > half_width) {
        out -= width;
    } else if (reference_col - out > half_width) {
        out += width;
    }
    return out;
}

void mark_proxy_pixel(
    std::vector<std::uint8_t>& mask,
    RiPixel pixel,
    ProxyRasterShape shape
)
{
    for (int dr = -shape.radius; dr <= shape.radius; ++dr) {
        const int rr = pixel.row + dr;
        if (rr < 0 || rr >= shape.height) {
            continue;
        }
        for (int dc = -shape.radius; dc <= shape.radius; ++dc) {
            int cc = pixel.col + dc;
            while (cc < 0) {
                cc += shape.width;
            }
            while (cc >= shape.width) {
                cc -= shape.width;
            }
            mask[static_cast<std::size_t>(rr * shape.width + cc)] = 1;
        }
    }
}

void mark_proxy_depth_pixel(
    std::vector<float>& depth,
    float range,
    RiPixel pixel,
    ProxyRasterShape shape
)
{
    if (!std::isfinite(range) || range <= 0.0F) {
        return;
    }
    for (int dr = -shape.radius; dr <= shape.radius; ++dr) {
        const int rr = pixel.row + dr;
        if (rr < 0 || rr >= shape.height) {
            continue;
        }
        for (int dc = -shape.radius; dc <= shape.radius; ++dc) {
            int cc = pixel.col + dc;
            while (cc < 0) {
                cc += shape.width;
            }
            while (cc >= shape.width) {
                cc -= shape.width;
            }
            float& value = depth[static_cast<std::size_t>(rr * shape.width + cc)];
            value = std::min(value, range);
        }
    }
}

bool mark_segment_proxy(
    ProxyRasterTarget target,
    ProjectedRiPoint a,
    ProjectedRiPoint b
)
{
    if (!a.valid || !b.valid) {
        return false;
    }
    const int width = target.shape.width;
    const int b_col = unwrap_col_near(a.col, b.col, width);
    const int drow = b.row - a.row;
    const int dcol = b_col - a.col;
    const int steps = std::max(std::abs(drow), std::abs(dcol));
    if (steps <= 0) {
        mark_proxy_pixel(target.mask, {a.row, a.col}, target.shape);
        mark_proxy_depth_pixel(target.depth, a.range, {a.row, a.col}, target.shape);
        return true;
    }
    for (int step = 0; step <= steps; ++step) {
        const MeshScalar t = static_cast<MeshScalar>(step) / static_cast<MeshScalar>(steps);
        const int row = static_cast<int>(std::llround(static_cast<MeshScalar>(a.row) + t * drow));
        int col = static_cast<int>(std::llround(static_cast<MeshScalar>(a.col) + t * dcol));
        while (col < 0) {
            col += width;
        }
        while (col >= width) {
            col -= width;
        }
        const float range = static_cast<float>(
            static_cast<MeshScalar>(a.range) + t * (static_cast<MeshScalar>(b.range) - static_cast<MeshScalar>(a.range))
        );
        mark_proxy_pixel(target.mask, {row, col}, target.shape);
        mark_proxy_depth_pixel(target.depth, range, {row, col}, target.shape);
    }
    return true;
}

bool mark_triangle_mask(
    std::vector<std::uint8_t>& mask,
    TriangleMaskRequest request
)
{
    const ProjectedRiPoint p0 = request.p0;
    const ProjectedRiPoint p1 = request.p1;
    const ProjectedRiPoint p2 = request.p2;
    const int height = request.height;
    const int width = request.width;
    if (!p0.valid || !p1.valid || !p2.valid) {
        return false;
    }
    const MeshScalar r0 = static_cast<MeshScalar>(p0.row);
    const MeshScalar c0 = static_cast<MeshScalar>(p0.col);
    const MeshScalar r1 = static_cast<MeshScalar>(p1.row);
    const MeshScalar c1 = static_cast<MeshScalar>(unwrap_col_near(p0.col, p1.col, width));
    const MeshScalar r2 = static_cast<MeshScalar>(p2.row);
    const MeshScalar c2 = static_cast<MeshScalar>(unwrap_col_near(p0.col, p2.col, width));
    const MeshScalar denom = (r1 - r2) * (c0 - c2) + (c2 - c1) * (r0 - r2);
    if (std::abs(denom) < static_cast<MeshScalar>(1e-9)) {
        return false;
    }
    const int row_min = std::max(0, static_cast<int>(std::floor(std::min({r0, r1, r2}))));
    const int row_max = std::min(height - 1, static_cast<int>(std::ceil(std::max({r0, r1, r2}))));
    const int col_min = static_cast<int>(std::floor(std::min({c0, c1, c2})));
    const int col_max = static_cast<int>(std::ceil(std::max({c0, c1, c2})));
    const std::int64_t bbox_pixels = static_cast<std::int64_t>(row_max - row_min + 1)
        * static_cast<std::int64_t>(col_max - col_min + 1);
    if (bbox_pixels <= 0 || bbox_pixels > 16384) {
        return false;
    }
    for (int row = row_min; row <= row_max; ++row) {
        for (int col_unwrapped = col_min; col_unwrapped <= col_max; ++col_unwrapped) {
            const MeshScalar row_d = static_cast<MeshScalar>(row);
            const MeshScalar col_d = static_cast<MeshScalar>(col_unwrapped);
            const MeshScalar w0 = ((r1 - r2) * (col_d - c2) + (c2 - c1) * (row_d - r2)) / denom;
            const MeshScalar w1 = ((r2 - r0) * (col_d - c2) + (c0 - c2) * (row_d - r2)) / denom;
            const MeshScalar w2 = static_cast<MeshScalar>(1) - w0 - w1;
            if (w0 < static_cast<MeshScalar>(-1e-6)
                || w1 < static_cast<MeshScalar>(-1e-6)
                || w2 < static_cast<MeshScalar>(-1e-6)) {
                continue;
            }
            int col = col_unwrapped;
            while (col < 0) {
                col += width;
            }
            while (col >= width) {
                col -= width;
            }
            mask[static_cast<std::size_t>(row * width + col)] = 1;
        }
    }
    return true;
}

std::size_t count_mask_pixels(const std::vector<std::uint8_t>& mask)
{
    return static_cast<std::size_t>(std::count(mask.begin(), mask.end(), std::uint8_t{1}));
}
StitchProxyScratch build_stitch_proxy_scratch(
    const StitchProxyBuildRequest& request
)
{
    const std::vector<Vec3f>& vertices_global = request.vertices_global;
    const std::vector<std::array<int, 2>>& old_boundary_edges = request.old_boundary_edges;
    const std::vector<std::array<int, 2>>& new_boundary_edges = request.new_boundary_edges;
    const std::vector<std::array<int, 2>>& old_new_edges = request.old_new_edges;
    const Mat4f& next_pose = request.next_pose;
    const SparseMeshingConfig& config = request.config;
    const bool collect_metrics = request.collect_metrics;
    const int height = static_cast<int>(config.ri_height);
    const int width = static_cast<int>(config.ri_width);
    const int pixel_count = height * width;
    const int proxy_radius = 0;
    const Mat4f next_global_to_local = inverse_rigid_cpu(next_pose);

    StitchProxyScratch scratch;
    scratch.metrics.evaluated = collect_metrics;
    scratch.metrics.old_new_candidate_edges = old_new_edges.size();
    scratch.proxy_mask.assign(static_cast<std::size_t>(pixel_count), std::uint8_t{0});
    scratch.proxy_depth.assign(
        static_cast<std::size_t>(pixel_count),
        std::numeric_limits<float>::infinity()
    );
    const auto draw_edge = [&](std::array<int, 2> edge) {
        if (edge[0] < 0 || edge[1] < 0
            || static_cast<std::size_t>(edge[0]) >= vertices_global.size()
            || static_cast<std::size_t>(edge[1]) >= vertices_global.size()) {
            return;
        }
        const ProjectedRiPoint a =
            project_global_to_ri(vertices_global[static_cast<std::size_t>(edge[0])], next_global_to_local, config);
        const ProjectedRiPoint b =
            project_global_to_ri(vertices_global[static_cast<std::size_t>(edge[1])], next_global_to_local, config);
        if (mark_segment_proxy(
            ProxyRasterTarget{scratch.proxy_mask, scratch.proxy_depth, {height, width, proxy_radius}}, a, b
        )) {
            if (collect_metrics) {
                ++scratch.metrics.proxy_segments;
            }
        }
    };
    for (const std::array<int, 2> edge : old_boundary_edges) {
        draw_edge(edge);
    }
    for (const std::array<int, 2> edge : new_boundary_edges) {
        draw_edge(edge);
    }
    for (const std::array<int, 2> edge : old_new_edges) {
        draw_edge(edge);
    }
    if (collect_metrics) {
        scratch.metrics.proxy_pixels = count_mask_pixels(scratch.proxy_mask);
    }
    return scratch;
}

void finish_stitch_proxy_metrics(const StitchProxyMetricsRequest& request)
{
    StitchProxyScratch& scratch = request.scratch;
    const std::vector<Vec3f>& vertices_global = request.vertices_global;
    const StitchProxyOracleFaces faces = request.faces;
    const Mat4f& next_pose = request.next_pose;
    const SparseMeshingConfig& config = request.config;
    if (!scratch.metrics.evaluated) {
        return;
    }
    const int height = static_cast<int>(config.ri_height);
    const int width = static_cast<int>(config.ri_width);
    const int pixel_count = height * width;
    const Mat4f next_global_to_local = inverse_rigid_cpu(next_pose);
    std::vector<std::uint8_t> oracle_mask(static_cast<std::size_t>(pixel_count), std::uint8_t{0});
    Stopwatch raster_oracle_stopwatch;
    const auto raster_faces = [&](const std::vector<Triangle>& source) {
        for (const Triangle face : source) {
            if (face.v0 < 0 || face.v1 < 0 || face.v2 < 0
                || static_cast<std::size_t>(face.v0) >= vertices_global.size()
                || static_cast<std::size_t>(face.v1) >= vertices_global.size()
                || static_cast<std::size_t>(face.v2) >= vertices_global.size()) {
                continue;
            }
            const ProjectedRiPoint p0 = project_global_to_ri(
                vertices_global[static_cast<std::size_t>(face.v0)], next_global_to_local, config
            );
            const ProjectedRiPoint p1 = project_global_to_ri(
                vertices_global[static_cast<std::size_t>(face.v1)], next_global_to_local, config
            );
            const ProjectedRiPoint p2 = project_global_to_ri(
                vertices_global[static_cast<std::size_t>(face.v2)], next_global_to_local, config
            );
            if (mark_triangle_mask(oracle_mask, TriangleMaskRequest{p0, p1, p2, height, width})) {
                ++scratch.metrics.oracle_faces;
            }
        }
    };
    raster_faces(faces.stitch_faces);
    raster_faces(faces.hole_faces);
    scratch.metrics.raster_oracle_milliseconds += raster_oracle_stopwatch.elapsed_milliseconds();

    Stopwatch compare_stopwatch;
    for (int idx = 0; idx < pixel_count; ++idx) {
        const bool proxy = scratch.proxy_mask[static_cast<std::size_t>(idx)] != 0;
        const bool oracle = oracle_mask[static_cast<std::size_t>(idx)] != 0;
        if (oracle) {
            ++scratch.metrics.oracle_pixels;
        }
        if (proxy && oracle) {
            ++scratch.metrics.intersection_pixels;
        } else if (!proxy && oracle) {
            ++scratch.metrics.false_negative_pixels;
        } else if (proxy && !oracle) {
            ++scratch.metrics.false_positive_pixels;
        }
    }
    scratch.metrics.compare_masks_milliseconds += compare_stopwatch.elapsed_milliseconds();
    if (request.error_map_dir != nullptr && !request.error_map_dir->empty()) {
        std::filesystem::create_directories(*request.error_map_dir);
        std::ofstream output(
            *request.error_map_dir
                / ("proxy_error_frame_" + std::to_string(request.frame_index) + ".pgm"),
            std::ios::binary
        );
        output << "P5\n" << width << ' ' << height << "\n3\n";
        for (int idx = 0; idx < pixel_count; ++idx) {
            const bool proxy = scratch.proxy_mask[static_cast<std::size_t>(idx)] != 0;
            const bool oracle = oracle_mask[static_cast<std::size_t>(idx)] != 0;
            const std::uint8_t category = proxy && oracle ? 1
                : proxy ? 2
                : oracle ? 3
                : 0;
            output.write(
                reinterpret_cast<const char*>(&category),
                static_cast<std::streamsize>(sizeof(category))
            );
        }
    }
}

}  // namespace ri_meshing::cuda_pipeline
