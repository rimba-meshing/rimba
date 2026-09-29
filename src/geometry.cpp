#include "ri_meshing/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ri_meshing {

MeshScalar point_range(Vec3f point) noexcept
{
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

std::optional<RiPixel> project_to_ri(Vec3f point, const SparseMeshingConfig& config)
{
    const MeshScalar xy = std::sqrt(point.x * point.x + point.y * point.y);
    const MeshScalar range = point_range(point);
    if (range <= 0.0) {
        return std::nullopt;
    }

    const MeshScalar phi = std::atan2(point.z, xy);
    const MeshScalar theta = std::atan2(point.y, point.x);
    const MeshScalar fov_range = config.vertical_fov.max_rad - config.vertical_fov.min_rad;
    const auto row = static_cast<int>(
        std::llround((phi - config.vertical_fov.min_rad) / fov_range * config.ri_height)
    );
    auto col = static_cast<int>(
        std::llround((theta + std::numbers::pi) / (2.0 * std::numbers::pi) * config.ri_width)
    );
    col = std::clamp(col, 0, static_cast<int>(config.ri_width) - 1);

    if (row < 0 || row >= static_cast<int>(config.ri_height)) {
        return std::nullopt;
    }
    return RiPixel{row, col, range};
}

TriangleMetrics triangle_metrics(Vec3f a, Vec3f b, Vec3f c) noexcept
{
    const Vec3f ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3f ac{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3f bc{c.x - b.x, c.y - b.y, c.z - b.z};

    const auto edge_sq = [](Vec3f edge) -> MeshScalar {
        return edge.x * edge.x + edge.y * edge.y + edge.z * edge.z;
    };
    const MeshScalar longest = std::max({edge_sq(ab), edge_sq(ac), edge_sq(bc)});
    const Vec3f cross{
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x,
    };
    const MeshScalar twice_area = point_range(cross);
    const MeshScalar quality = longest <= 0.0F ? -1.0F : twice_area / longest;
    return TriangleMetrics{longest, twice_area, quality};
}

bool triangle_passes_filters(Vec3f a, Vec3f b, Vec3f c, const SparseMeshingConfig& config) noexcept
{
    const TriangleMetrics metrics = triangle_metrics(a, b, c);
    const MeshScalar max_edge_sq = config.max_triangle_dist * config.max_triangle_dist;
    return metrics.longest_edge_sq <= max_edge_sq
        && static_cast<MeshScalar>(0.5) * metrics.twice_area >= config.min_triangle_area
        && metrics.quality >= config.min_triangle_quality;
}

}  // namespace ri_meshing
