#pragma once

#include "stitch_state.hpp"
#include "ri_meshing/config.hpp"

#include <filesystem>

namespace ri_meshing::cuda_pipeline {

struct StitchProxyBuildRequest {
    const std::vector<Vec3f>& vertices_global;
    const std::vector<std::array<int, 2>>& old_boundary_edges;
    const std::vector<std::array<int, 2>>& new_boundary_edges;
    const std::vector<std::array<int, 2>>& old_new_edges;
    Mat4f next_pose{};
    const SparseMeshingConfig& config;
    bool collect_metrics{};
};

ProjectedRiPoint project_global_to_ri(
    Vec3f point_global,
    const Mat4f& global_to_local,
    const SparseMeshingConfig& config
);

StitchProxyScratch build_stitch_proxy_scratch(const StitchProxyBuildRequest& request);
struct StitchProxyOracleFaces {
    const std::vector<Triangle>& stitch_faces;
    const std::vector<Triangle>& hole_faces;
};

struct StitchProxyMetricsRequest {
    StitchProxyScratch& scratch;
    const std::vector<Vec3f>& vertices_global;
    StitchProxyOracleFaces faces;
    Mat4f next_pose{};
    const SparseMeshingConfig& config;
    const std::filesystem::path* error_map_dir{};
    int frame_index{-1};
};

void finish_stitch_proxy_metrics(const StitchProxyMetricsRequest& request);

}  // namespace ri_meshing::cuda_pipeline
