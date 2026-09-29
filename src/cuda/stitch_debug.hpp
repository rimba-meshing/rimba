#pragma once

#include "ri_meshing/boundary_stitching.hpp"
#include "ri_meshing/cuda_meshing.hpp"
#include "ri_meshing/hole_filling.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace ri_meshing::cuda_pipeline {

struct StitchFrameDebugSnapshot {
    int frame_index{};
    std::string label;
    std::vector<Vec3f> vertices_global;
    std::vector<Triangle> faces_before_stitch;
    std::vector<Triangle> new_faces;
    std::vector<Triangle> stitch_faces;
    std::vector<Triangle> hole_faces;
    StitchCoreDebug core;
    std::size_t old_face_count{};
    std::size_t new_face_count{};
};

void remap_stitch_core_debug(StitchCoreDebug& debug, const std::vector<int>& compact_to_global);
void dump_stitch_debug_snapshot(const StitchDebugDump& debug_dump, const StitchFrameDebugSnapshot& snapshot);
void dump_hole_fill_metrics(
    const HoleFillMetricsDump& debug_dump,
    int frame_index,
    const HoleFillResult& result
);

}  // namespace ri_meshing::cuda_pipeline
