#pragma once

#include "ri_meshing/cuda_meshing.hpp"
#include "ri_meshing/boundary_stitching.hpp"
#include "ri_meshing/config.hpp"
#include "ri_meshing/timing.hpp"
#include "ri_meshing/types.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace ri_meshing::cuda_detail {

using StageTimelineCallback = std::function<void(const char*)>;

struct FaceRecord {
    int key0{};
    int key1{};
    int key2{};
    Triangle face{};
};

struct RiMacroQuad {
    int row{};
    int col{};
    int row_size{};
    int col_size{};
    // A value of 1 means the planner already validated the two corner
    // triangles; 0 denotes a dense 1x1 fallback cell.
    unsigned char corner_triangles{};
};

struct AccumulatedDepthView {
    const float* values{};
    bool bottom_up_normalized{};
};

struct AccumulatedFrameChunk {
    // The index ranges describe all geometry appended by this frame. The
    // spatial bound is deliberately computed from the dense coarsened mesh
    // only; post-dense geometry stays inside this cheap frame-level
    // approximation without expanding its bound.
    int first_vertex{};
    int vertex_count{};
    int first_face{};
    int face_count{};
    Vec3f center_global{};
    float radius{};
};

struct AccumulatedFaceRange {
    int first_face{};
    int face_count{};
    int first_active{};
};

struct AccumulatedVertexRange {
    int first_vertex{};
    int vertex_count{};
    int first_active{};
};

inline bool accumulated_chunk_intersects_radius(
    const AccumulatedFrameChunk& chunk,
    Vec3f sensor_origin_global,
    float query_radius
)
{
    const float dx = chunk.center_global.x - sensor_origin_global.x;
    const float dy = chunk.center_global.y - sensor_origin_global.y;
    const float dz = chunk.center_global.z - sensor_origin_global.z;
    const float limit = chunk.radius + query_radius;
    return dx * dx + dy * dy + dz * dz <= limit * limit;
}

struct CudaWorkspace {
    thrust::device_vector<Vec3f> points;
    thrust::device_vector<Vec3f> accumulated_vertices_global;
    thrust::device_vector<Vec3f> current_vertex_normal_sums;
    thrust::device_vector<Triangle> accumulated_faces_global;
    thrust::device_vector<Vec3f> current_vertices_local;
    thrust::device_vector<float> accumulated_depth;
    thrust::device_vector<int> projected_rows;
    thrust::device_vector<int> projected_cols;
    thrust::device_vector<float> projected_ranges;
    thrust::device_vector<int> pixel_last_index;
    thrust::device_vector<unsigned char> valid;
    thrust::device_vector<unsigned char> clean_valid;
    thrust::device_vector<unsigned char> meshing_valid;
    thrust::device_vector<unsigned char> new_vertex_allowed;
    thrust::device_vector<unsigned char> final_new_allowed;
    thrust::device_vector<unsigned char> ri_thin_input;
    thrust::device_vector<unsigned char> ri_thin_eroded;
    thrust::device_vector<unsigned char> ri_thin_grown;
    thrust::device_vector<unsigned char> ri_thin_removed;
    thrust::device_vector<unsigned char> stitch_connection_mask;
    thrust::device_vector<unsigned char> stitch_connection_old_support;
    thrust::device_vector<unsigned char> stitch_connection_new_support;
    thrust::device_vector<unsigned char> stitch_connection_next_frontier;
    StitchConnectionMask host_stitch_connection_mask;
    thrust::device_vector<Vec3f> pixel_xyz;
    thrust::device_vector<float> scan_ranges;
    thrust::device_vector<float> filled_ranges;
    thrust::device_vector<float> fill_scratch_ranges;
    thrust::device_vector<float> fill_support_ranges;
    thrust::device_vector<unsigned char> fill_anchor;
    thrust::device_vector<unsigned char> fill_scratch_anchor;
    thrust::device_vector<Vec3f> compacted_current_vertices;
    thrust::device_vector<Vec3f> filled_pixel_xyz;
    thrust::device_vector<int> current_vertex_rows;
    thrust::device_vector<int> current_vertex_cols;
    thrust::device_vector<float> current_vertex_ranges;
    thrust::device_vector<int> flags;
    thrust::device_vector<int> offsets;
    thrust::device_vector<int> large_face_indices;
    thrust::device_vector<int> filled_vim;
    thrust::device_vector<RiMacroQuad> macro_quads;
    thrust::device_vector<int> macro_quad_count;
    thrust::device_vector<unsigned char> macro_candidate_flags;
    thrust::device_vector<unsigned char> macro_selected_size;
    thrust::device_vector<unsigned char> macro_next_size;
    thrust::device_vector<FaceRecord> face_records;
    thrust::device_vector<int> face_record_count;
    thrust::device_vector<int> face_overflow;
    thrust::device_vector<unsigned char> quad_face_flags;
    thrust::device_vector<Triangle> faces;
    thrust::device_vector<std::uint64_t> edge_keys;
    thrust::device_vector<std::uint64_t> oriented_boundary_edges;
    thrust::device_vector<std::uint64_t> boundary_edges;
    thrust::device_vector<AccumulatedFaceRange> active_face_ranges;
    thrust::device_vector<AccumulatedVertexRange> active_vertex_ranges;
    std::vector<std::uint64_t> host_boundary_edges;
};

struct CudaRiFrame {
    int height{};
    int width{};
    int old_vertex_count{};
    int new_vertex_count{};
    thrust::device_vector<Vec3f>* vertices{};
    thrust::device_vector<int>* filled_vim{};
    thrust::device_vector<RiMacroQuad>* macro_quads{};
};

struct CudaPixelScanPrepass {
    int height{};
    int width{};
    int pixel_count{};
};

struct CudaAccumulationCounts {
    int vertex_count{};
    int face_count{};
};

struct CudaExecutionContext {
    CudaWorkspace& workspace;
    PipelineTiming* timing{};
    cudaStream_t stream{};
    StageTimelineCallback timeline;
};

struct ProjectedVertexSupportView {
    const int* rows{};
    const int* cols{};
    const float* ranges{};
    int vertex_count{};
};

struct ScanBuildOptions {
    int old_vertex_count{};
    const AccumulatedDepthView* accumulated_depth{};
    const AccumulatedDepthView* accumulated_fill_support{};
    bool track_fill_anchors{};
    ProjectedVertexSupportView projected_vertex_support{};
};

struct DenseMeshOptions {
    const AccumulatedDepthView* accumulated_depth{};
};

struct AccumulatedDepthOptions {
    CudaAccumulationCounts counts;
    Mat4f pose;
    const std::vector<AccumulatedFrameChunk>* frame_chunks{};
};

struct AccumulatedDepthAppendOptions {
    CudaAccumulationCounts counts;
    Mat4f pose;
    int first_vertex{};
    int vertex_count{};
    int first_face{};
    int face_count{};
};

struct AccumulationAppendInput {
    const CudaRiFrame& frame;
    const thrust::device_vector<Triangle>& faces;
    const Mat4f& pose;
};

CudaRiFrame scan_and_build_pixel_vertices_cuda(
    const std::vector<Vec3f>& points,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanBuildOptions options = {}
);

CudaPixelScanPrepass scan_pixel_vertices_pre_blocker_cuda(
    const std::vector<Vec3f>& points,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanBuildOptions options = {}
);

CudaRiFrame finish_pixel_vertices_from_pre_scan_cuda(
    CudaPixelScanPrepass prepass,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanBuildOptions options = {}
);

void apply_accumulated_fill_support_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    CudaPixelScanPrepass prepass,
    const ScanBuildOptions& options
);

thrust::device_vector<Triangle>& coarsened_dense_mesh_cuda(
    const CudaRiFrame& frame,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    DenseMeshOptions options
);

void prepare_accumulated_depth_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    AccumulatedDepthOptions options
);

void append_accumulated_depth_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    AccumulatedDepthAppendOptions options
);

void append_current_frame_to_accumulation_cuda(
    CudaAccumulationCounts& counts,
    CudaExecutionContext execution,
    AccumulationAppendInput input
);

void compact_current_frame_vertices_cuda(
    CudaRiFrame& frame,
    thrust::device_vector<Triangle>& faces,
    CudaExecutionContext execution
);

void rebase_prefetched_frame_cuda(
    CudaRiFrame& frame,
    thrust::device_vector<Triangle>& faces,
    int new_old_vertex_count,
    CudaExecutionContext execution
);

}  // namespace ri_meshing::cuda_detail
