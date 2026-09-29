#pragma once

#include "ri_meshing/config.hpp"
#include "ri_meshing/types.hpp"

#include <optional>

namespace ri_meshing {

struct RiPixel {
    int row{};
    int col{};
    MeshScalar range{};
};

struct TriangleMetrics {
    MeshScalar longest_edge_sq{};
    MeshScalar twice_area{};
    MeshScalar quality{};
};

[[nodiscard]] MeshScalar point_range(Vec3f point) noexcept;
[[nodiscard]] std::optional<RiPixel> project_to_ri(Vec3f point, const SparseMeshingConfig& config);
[[nodiscard]] TriangleMetrics triangle_metrics(Vec3f a, Vec3f b, Vec3f c) noexcept;
[[nodiscard]] bool triangle_passes_filters(
    Vec3f a,
    Vec3f b,
    Vec3f c,
    const SparseMeshingConfig& config
) noexcept;

}  // namespace ri_meshing
