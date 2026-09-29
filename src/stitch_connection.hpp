#pragma once

#include "ri_meshing/boundary_paths.hpp"
#include "ri_meshing/boundary_stitching.hpp"
#include "ri_meshing/config.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace ri_meshing::stitch_detail {

struct StitchConnectionOracleRequest {
    const StitchConnectionMask& mask;
    const SparseMeshingConfig& config;
    const std::vector<Vec3f>& vertices;
    const std::vector<BoundaryPath>& old_paths;
    const std::vector<BoundaryPath>& new_paths;
    PipelineTiming* timing{};
};

class StitchConnectionOracle {
public:
    explicit StitchConnectionOracle(const StitchConnectionOracleRequest& request);

    // Reuse the exact raw projection prepared for this oracle. Support
    // anchors are intentionally kept separate from this query.
    [[nodiscard]] std::array<int, 2> projected_pixel(int vertex) const;

    // Keeps later pair and face checks consistent with a validated local gap repair.
    void inherit_component_support(int vertex, int component);
    // Label matching enforces the red-component step limit.
    [[nodiscard]] int component_for_pair(int old_vertex, int new_vertex) const;
    // Finds the first red component reached by an angular RI-space arrow.
    [[nodiscard]] int component_for_open_side(
        int vertex,
        MeshScalar angular_x,
        MeshScalar angular_y,
        bool source_is_old
    ) const;
    [[nodiscard]] bool pair_allowed(int old_vertex, int new_vertex) const;
    // Post-label geometry checks keep the frontier-component constraint but do
    // not reapply the RI jump limit already used during label matching.
    [[nodiscard]] bool pair_allowed_for_geometry(int old_vertex, int new_vertex) const;
    [[nodiscard]] bool face_allowed(Triangle face, int old_vertex_count) const;

private:
    struct Pixel {
        int row{-1};
        int col{-1};
    };

    struct ComponentSupport {
        int component{-1};
        Pixel anchor{};
        Pixel vertex_pixel{};
        // This neighborhood is immutable for one oracle and is reused across
        // support candidates for the same projected vertex.
        int nearest_empty_distance_squared{-1};
    };

    [[nodiscard]] Pixel project(Vec3f point) const;
    [[nodiscard]] Pixel projected_pixel_for_vertex(int vertex) const;
    [[nodiscard]] int wrap_column(int col) const;
    [[nodiscard]] bool valid_pixel(Pixel pixel) const;
    [[nodiscard]] int pixel_chebyshev_distance(Pixel first, Pixel second) const;
    [[nodiscard]] int pixel_distance_squared(Pixel first, Pixel second) const;
    [[nodiscard]] bool empty_pixel(Pixel pixel) const;
    [[nodiscard]] int compute_nearest_empty_distance_squared(Pixel start) const;
    [[nodiscard]] int nearest_empty_distance_squared(int vertex, Pixel start) const;
    [[nodiscard]] bool old_support_pixel(Pixel pixel) const;
    [[nodiscard]] bool new_support_pixel(Pixel pixel) const;
    [[nodiscard]] bool segment_supported_by_old(Pixel first, Pixel second) const;
    [[nodiscard]] Pixel nearest_component_pixel(Pixel start, int component) const;
    [[nodiscard]] bool within_component_red_steps(
        Pixel first,
        Pixel second,
        int component
    ) const;
    [[nodiscard]] bool within_component_red_steps_fast(
        Pixel first,
        Pixel second,
        int component
    ) const;
    [[nodiscard]] int component_for_pair_impl(
        int old_vertex,
        int new_vertex,
        bool enforce_red_steps
    ) const;
    [[nodiscard]] bool segment_touches_component(Pixel first, Pixel second, int component) const;
    void build_components();
    void build_linear_support_offsets();
    void add_boundary_support(const std::vector<BoundaryPath>& paths);
    void prepare_nearest_empty_cache(
        const std::vector<BoundaryPath>& old_paths,
        const std::vector<BoundaryPath>& new_paths
    );
    void add_edge_support(
        int first,
        int second,
        std::vector<ComponentSupport>& edge_support_scratch
    );
    void add_linear_frontier_support(int vertex);
    void add_vertex_component(int vertex, int component, Pixel anchor);
    void collect_nearby_supports(Pixel pixel, std::vector<ComponentSupport>& supports) const;

    const StitchConnectionMask& mask_;
    const SparseMeshingConfig& config_;
    const std::vector<Vec3f>& vertices_;
    PipelineTiming* timing_{};
    std::size_t pixel_count_{};
    bool mask_pixels_available_{};
    bool old_support_available_{};
    bool new_support_available_{};
    std::vector<int> component_ids_;
    std::vector<Pixel> projected_pixels_;
    // Boundary vertex IDs are compact and dense. Cache only queried vertices,
    // rather than reserving one entry for every pixel in the full RI.
    mutable std::vector<int> nearest_empty_by_vertex_;
    // Stable distance order preserves the old row-major tie break.
    std::vector<Pixel> linear_support_offsets_;
    // Stitch-core vertex IDs are compact and dense. Keep one slot per ID so
    // hot pair checks do not pay for hashing or repeated RI projection.
    std::vector<ComponentSupport> vertex_components_;
};

}  // namespace ri_meshing::stitch_detail
