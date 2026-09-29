#include "cuda_workflow.hpp"

#include "accumulation_transform.cuh"
#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace ri_meshing::cuda_detail {
namespace {

constexpr float kPiF = 3.1415926535897932384626433832795F;
constexpr int kLargeRasterFacePixelThreshold = 4096;

struct DeviceMat4f {
    float m[16];
};

struct RiGeometry {
    int height;
    int width;
};

struct ProjectionMap {
    int* rows;
    int* cols;
    float* ranges;
};

struct ConstProjectionMap {
    const int* rows;
    const int* cols;
    const float* ranges;
};

struct ProjectionBuffers {
    const Vec3f* vertices;
    ProjectionMap projection;
    const AccumulatedVertexRange* active_ranges;
    int active_range_count;
};

struct FastProjectionParams {
    DeviceMat4f transform;
    RiGeometry geometry;
    float fov_min;
    float fov_range;
    int count;
};

template <typename Scalar>
struct ProjectedFaceBounds {
    Scalar r0;
    Scalar c0;
    Scalar r1;
    Scalar c1;
    Scalar r2;
    Scalar c2;
    int row_min;
    int row_max;
    int col_min;
    int col_max;
};

template <typename Scalar>
struct BarycentricTriangle {
    Scalar r0;
    Scalar c0;
    Scalar r1;
    Scalar c1;
    Scalar r2;
    Scalar c2;
    Scalar denom;
};

template <typename Scalar>
struct BarycentricWeights {
    Scalar w0;
    Scalar w1;
    Scalar w2;
};

struct FaceRanges {
    float range0;
    float range1;
    float range2;
};

struct PixelCoordinate {
    int row;
    int col;
};

struct DepthTarget {
    float* depth;
    int width;
};

struct ProjectedMeshView {
    const Triangle* faces;
    ConstProjectionMap projection;
    const AccumulatedFaceRange* active_ranges;
    int active_range_count;
    int face_count;
};

struct FaceRasterView {
    ProjectedMeshView mesh;
    DepthTarget target;
    RiGeometry geometry;
};

struct LargeFaceView {
    FaceRasterView raster;
    const int* indices;
    int count;
};

static_assert(std::is_trivially_copyable_v<RiGeometry>);
static_assert(std::is_trivially_copyable_v<ProjectionMap>);
static_assert(std::is_trivially_copyable_v<ConstProjectionMap>);
static_assert(std::is_trivially_copyable_v<ProjectionBuffers>);
static_assert(std::is_trivially_copyable_v<FastProjectionParams>);
static_assert(std::is_trivially_copyable_v<ProjectedFaceBounds<float>>);
static_assert(std::is_trivially_copyable_v<BarycentricTriangle<float>>);
static_assert(std::is_trivially_copyable_v<BarycentricWeights<float>>);
static_assert(std::is_trivially_copyable_v<FaceRanges>);
static_assert(std::is_trivially_copyable_v<PixelCoordinate>);
static_assert(std::is_trivially_copyable_v<DepthTarget>);
static_assert(std::is_trivially_copyable_v<ProjectedMeshView>);
static_assert(std::is_trivially_copyable_v<FaceRasterView>);
static_assert(std::is_trivially_copyable_v<LargeFaceView>);

__device__ int active_face_index(ProjectedMeshView mesh, int active_index)
{
    if (mesh.active_ranges == nullptr || mesh.active_range_count == 0) {
        return active_index;
    }
    int lower = 0;
    int upper = mesh.active_range_count;
    while (lower < upper) {
        const int middle = lower + (upper - lower) / 2;
        const AccumulatedFaceRange range = mesh.active_ranges[middle];
        if (active_index < range.first_active) {
            upper = middle;
        } else if (active_index >= range.first_active + range.face_count) {
            lower = middle + 1;
        } else {
            return range.first_face + active_index - range.first_active;
        }
    }
    return -1;
}

__device__ int active_vertex_index(ProjectionBuffers buffers, int active_index)
{
    if (buffers.active_ranges == nullptr || buffers.active_range_count == 0) {
        return active_index;
    }
    int lower = 0;
    int upper = buffers.active_range_count;
    while (lower < upper) {
        const int middle = lower + (upper - lower) / 2;
        const AccumulatedVertexRange range = buffers.active_ranges[middle];
        if (active_index < range.first_active) {
            upper = middle;
        } else if (active_index >= range.first_active + range.vertex_count) {
            lower = middle + 1;
        } else {
            return range.first_vertex + active_index - range.first_active;
        }
    }
    return -1;
}

DeviceMat4f to_device_mat_f(Mat4f mat)
{
    DeviceMat4f out{};
    for (int idx = 0; idx < 16; ++idx) {
        out.m[idx] = static_cast<float>(mat[idx]);
    }
    return out;
}

Mat4f inverse_rigid(Mat4f pose)
{
    Mat4f out{1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out[row * 4 + col] = pose[col * 4 + row];
        }
    }
    for (int row = 0; row < 3; ++row) {
        out[row * 4 + 3] = -(
            out[row * 4 + 0] * pose[3]
            + out[row * 4 + 1] * pose[7]
            + out[row * 4 + 2] * pose[11]
        );
    }
    return out;
}

__global__ void transform_and_project_vertices_fast_kernel(
    ProjectionBuffers buffers,
    FastProjectionParams params
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.count) {
        return;
    }
    const int vertex_idx = active_vertex_index(buffers, idx);
    if (vertex_idx < 0) {
        return;
    }
    const Vec3f point = buffers.vertices[vertex_idx];
    const auto& transform = params.transform;
    const float x = transform.m[0] * point.x + transform.m[1] * point.y + transform.m[2] * point.z + transform.m[3];
    const float y = transform.m[4] * point.x + transform.m[5] * point.y + transform.m[6] * point.z + transform.m[7];
    const float z = transform.m[8] * point.x + transform.m[9] * point.y + transform.m[10] * point.z + transform.m[11];
    const float w = transform.m[12] * point.x + transform.m[13] * point.y + transform.m[14] * point.z + transform.m[15];
    const float inv_w = fabsf(w) > 1.0e-12F ? 1.0F / w : 1.0F;
    const float local_x = x * inv_w;
    const float local_y = y * inv_w;
    const float local_z = z * inv_w;

    const float xy = hypotf(local_x, local_y);
    const float range = hypotf(xy, local_z);
    if (range <= 1.0e-12F) {
        buffers.projection.rows[vertex_idx] = -1;
        buffers.projection.cols[vertex_idx] = -1;
        buffers.projection.ranges[vertex_idx] = 0.0F;
        return;
    }
    const float phi = atan2f(local_z, xy);
    const float theta = atan2f(local_y, local_x);
    const int row = static_cast<int>(lroundf(
        (phi - params.fov_min) / params.fov_range * static_cast<float>(params.geometry.height)
    ));
    int col = static_cast<int>(lroundf(
        (theta + kPiF) / (2.0F * kPiF) * static_cast<float>(params.geometry.width)
    ));
    col = min(max(col, 0), params.geometry.width - 1);
    if (row < 0 || row >= params.geometry.height) {
        buffers.projection.rows[vertex_idx] = -1;
        buffers.projection.cols[vertex_idx] = -1;
        buffers.projection.ranges[vertex_idx] = 0.0F;
        return;
    }
    buffers.projection.rows[vertex_idx] = row;
    buffers.projection.cols[vertex_idx] = col;
    buffers.projection.ranges[vertex_idx] = range;
}

__device__ bool projected_face_bounds_f(
    Triangle face,
    ConstProjectionMap projection,
    RiGeometry geometry,
    ProjectedFaceBounds<float>* bounds
)
{
    if (face.v0 < 0 || face.v1 < 0 || face.v2 < 0) {
        return false;
    }
    if (projection.rows[face.v0] < 0 || projection.rows[face.v1] < 0 || projection.rows[face.v2] < 0) {
        return false;
    }

    float r0 = static_cast<float>(projection.rows[face.v0]);
    float c0 = static_cast<float>(projection.cols[face.v0]);
    float r1 = static_cast<float>(projection.rows[face.v1]);
    float c1 = static_cast<float>(projection.cols[face.v1]);
    float r2 = static_cast<float>(projection.rows[face.v2]);
    float c2 = static_cast<float>(projection.cols[face.v2]);
    const float width_f = static_cast<float>(geometry.width);
    if (fmaxf(c0, fmaxf(c1, c2)) - fminf(c0, fminf(c1, c2)) > width_f * 0.5F) {
        if (c0 < width_f * 0.5F) {
            c0 += width_f;
        }
        if (c1 < width_f * 0.5F) {
            c1 += width_f;
        }
        if (c2 < width_f * 0.5F) {
            c2 += width_f;
        }
    }

    const int row_min = max(0, static_cast<int>(floorf(fminf(r0, fminf(r1, r2)))));
    const int row_max = min(geometry.height - 1, static_cast<int>(ceilf(fmaxf(r0, fmaxf(r1, r2)))));
    if (row_min > row_max) {
        return false;
    }

    *bounds = {r0, c0, r1, c1, r2, c2, row_min, row_max,
               static_cast<int>(floorf(fminf(c0, fminf(c1, c2)))),
               static_cast<int>(ceilf(fmaxf(c0, fmaxf(c1, c2))))};
    return true;
}

__device__ void atomic_min_positive_float(float* address, float value)
{
    atomicMin(reinterpret_cast<int*>(address), __float_as_int(value));
}

__device__ bool barycentric_with_denom_f(
    BarycentricTriangle<float> triangle,
    float row,
    float col,
    BarycentricWeights<float>* weights
)
{
    const float w0 = ((triangle.c1 - triangle.c2) * (row - triangle.r2)
                      + (triangle.r2 - triangle.r1) * (col - triangle.c2)) / triangle.denom;
    const float w1 = ((triangle.c2 - triangle.c0) * (row - triangle.r2)
                      + (triangle.r0 - triangle.r2) * (col - triangle.c2)) / triangle.denom;
    const float w2 = 1.0F - w0 - w1;
    if (w0 < 0.0F || w1 < 0.0F || w2 < 0.0F) {
        return false;
    }
    *weights = {w0, w1, w2};
    return true;
}

__device__ void rasterize_candidate_pixel_fast(
    BarycentricTriangle<float> triangle,
    FaceRanges ranges,
    PixelCoordinate pixel,
    DepthTarget target
)
{
    BarycentricWeights<float> weights{};
    if (!barycentric_with_denom_f(triangle, static_cast<float>(pixel.row), static_cast<float>(pixel.col), &weights)) {
        return;
    }
    int wrapped_col = pixel.col % target.width;
    if (wrapped_col < 0) {
        wrapped_col += target.width;
    }
    const float range = weights.w0 * ranges.range0 + weights.w1 * ranges.range1 + weights.w2 * ranges.range2;
    atomic_min_positive_float(&target.depth[pixel.row * target.width + wrapped_col], range);
}

__global__ void mark_large_raster_faces_kernel(
    ProjectedMeshView mesh,
    int* flags,
    RiGeometry geometry,
    int threshold
)
{
    const int active_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_index >= mesh.face_count) {
        return;
    }
    const int face_idx = active_face_index(mesh, active_index);
    if (face_idx < 0) {
        flags[active_index] = 0;
        return;
    }
    ProjectedFaceBounds<float> bounds{};
    if (!projected_face_bounds_f(mesh.faces[face_idx], mesh.projection, geometry, &bounds)) {
        flags[active_index] = 0;
        return;
    }
    const long long pixel_count =
        static_cast<long long>(bounds.row_max - bounds.row_min + 1)
        * static_cast<long long>(bounds.col_max - bounds.col_min + 1);
    flags[active_index] = pixel_count > threshold ? 1 : 0;
}

__global__ void compact_large_face_indices_kernel(
    const int* flags,
    const int* offsets,
    int* large_face_indices,
    int face_count
)
{
    const int active_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_index < face_count && flags[active_index] != 0) {
        large_face_indices[offsets[active_index]] = active_index;
    }
}

__global__ void rasterize_accumulated_faces_fast_kernel(
    FaceRasterView raster
)
{
    const int active_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_index >= raster.mesh.face_count) {
        return;
    }
    const int face_idx = active_face_index(raster.mesh, active_index);
    if (face_idx < 0) {
        return;
    }
    const Triangle face = raster.mesh.faces[face_idx];
    ProjectedFaceBounds<float> bounds{};
    if (!projected_face_bounds_f(face, raster.mesh.projection, raster.geometry, &bounds)) {
        return;
    }
    const int row_count = bounds.row_max - bounds.row_min + 1;
    const int col_count = bounds.col_max - bounds.col_min + 1;
    if (row_count * col_count > kLargeRasterFacePixelThreshold) {
        return;
    }
    const BarycentricTriangle<float> triangle{
        bounds.r0, bounds.c0, bounds.r1, bounds.c1, bounds.r2, bounds.c2,
        (bounds.c1 - bounds.c2) * (bounds.r0 - bounds.r2)
            + (bounds.r2 - bounds.r1) * (bounds.c0 - bounds.c2)
    };
    if (triangle.denom == 0.0F) {
        return;
    }
    const FaceRanges ranges{
        raster.mesh.projection.ranges[face.v0],
        raster.mesh.projection.ranges[face.v1],
        raster.mesh.projection.ranges[face.v2]
    };

    for (int row = bounds.row_min; row <= bounds.row_max; ++row) {
        for (int col = bounds.col_min; col <= bounds.col_max; ++col) {
            rasterize_candidate_pixel_fast(triangle, ranges, {row, col}, raster.target);
        }
    }
}

__global__ void rasterize_large_accumulated_faces_fast_kernel(
    LargeFaceView large_faces
)
{
    const int large_idx = blockIdx.x;
    if (large_idx >= large_faces.count) {
        return;
    }
    const auto raster = large_faces.raster;
    const int active_index = large_faces.indices[large_idx];
    const int face_idx = active_face_index(raster.mesh, active_index);
    if (face_idx < 0) {
        return;
    }
    const Triangle face = raster.mesh.faces[face_idx];
    ProjectedFaceBounds<float> bounds{};
    if (!projected_face_bounds_f(face, raster.mesh.projection, raster.geometry, &bounds)) {
        return;
    }

    const int col_count = bounds.col_max - bounds.col_min + 1;
    const int pixel_count = (bounds.row_max - bounds.row_min + 1) * col_count;
    const BarycentricTriangle<float> triangle{
        bounds.r0, bounds.c0, bounds.r1, bounds.c1, bounds.r2, bounds.c2,
        (bounds.c1 - bounds.c2) * (bounds.r0 - bounds.r2)
            + (bounds.r2 - bounds.r1) * (bounds.c0 - bounds.c2)
    };
    if (triangle.denom == 0.0F) {
        return;
    }
    const FaceRanges ranges{
        raster.mesh.projection.ranges[face.v0],
        raster.mesh.projection.ranges[face.v1],
        raster.mesh.projection.ranges[face.v2]
    };
    for (int pixel_offset = threadIdx.x; pixel_offset < pixel_count; pixel_offset += blockDim.x) {
        const int row = bounds.row_min + pixel_offset / col_count;
        const int col = bounds.col_min + pixel_offset - (pixel_offset / col_count) * col_count;
        rasterize_candidate_pixel_fast(triangle, ranges, {row, col}, raster.target);
    }
}

struct AccumulatedDepthRequest {
    CudaWorkspace* workspace;
    CudaAccumulationCounts counts;
    const Mat4f* pose;
    const std::vector<AccumulatedFrameChunk>* frame_chunks;
    const SparseMeshingConfig* config;
    PipelineTiming* timing;
    cudaStream_t stream;
    bool append_only{};
    int first_vertex{};
    int vertex_count{};
    int first_face{};
    int face_count{};
};

class AccumulatedDepthPass {
public:
    explicit AccumulatedDepthPass(AccumulatedDepthRequest request)
        : request_(request),
          geometry_{static_cast<int>(request.config->ri_height), static_cast<int>(request.config->ri_width)},
          global_to_local_fast_(to_device_mat_f(inverse_rigid(*request.pose)))
    {
    }

    void run()
    {
        prepare_buffers();
        if (!request_.append_only && request_.counts.vertex_count == 0) {
            return;
        }
        if (request_.append_only) {
            prepare_append_ranges();
        } else {
            prepare_active_face_ranges();
            clear_depth();
        }
        project_vertices();
        if (active_face_count_ == 0) {
            return;
        }
        rasterize_split_by_size();
    }

private:
    static constexpr int kBlockSize = 256;

    static int grid_for(int count)
    {
        return (count + kBlockSize - 1) / kBlockSize;
    }

    CudaWorkspace& workspace() const
    {
        return *request_.workspace;
    }

    const SparseMeshingConfig& config() const
    {
        return *request_.config;
    }

    void prepare_buffers()
    {
        timed_cuda_stage_void(request_.timing, "accum.prepare_buffers", [&] {
            workspace().current_vertices_local.clear();
        });
    }

    void prepare_active_face_ranges()
    {
        active_face_count_ = request_.counts.face_count;
        active_vertex_count_ = request_.counts.vertex_count;
        if (request_.counts.vertex_count <= 0 || request_.frame_chunks == nullptr
            || request_.frame_chunks->empty()) {
            workspace().active_face_ranges.clear();
            workspace().active_vertex_ranges.clear();
            return;
        }

        const Vec3f sensor_origin{
            request_.pose->at(3), request_.pose->at(7), request_.pose->at(11)
        };
        const float query_radius = static_cast<float>(config().accumulated_depth_cull_radius);
        std::vector<AccumulatedFaceRange> host_face_ranges;
        std::vector<AccumulatedVertexRange> host_vertex_ranges;
        host_face_ranges.reserve(request_.frame_chunks->size());
        host_vertex_ranges.reserve(request_.frame_chunks->size());
        active_face_count_ = 0;
        active_vertex_count_ = 0;
        for (const AccumulatedFrameChunk& chunk : *request_.frame_chunks) {
            if (!accumulated_chunk_intersects_radius(chunk, sensor_origin, query_radius)) {
                continue;
            }
            append_active_face_range(host_face_ranges, chunk, request_.counts.face_count);
            append_active_vertex_range(host_vertex_ranges, chunk, request_.counts.vertex_count);
        }
        upload_active_face_ranges(host_face_ranges);
        upload_active_vertex_ranges(host_vertex_ranges);
        active_face_count_ = range_count(host_face_ranges);
        active_vertex_count_ = vertex_range_count(host_vertex_ranges);
    }

    void prepare_append_ranges()
    {
        active_face_count_ = 0;
        active_vertex_count_ = 0;
        std::vector<AccumulatedFaceRange> face_ranges;
        std::vector<AccumulatedVertexRange> vertex_ranges;
        if (request_.face_count > 0) {
            const int first_face = std::max(0, request_.first_face);
            const int face_end = std::min(
                request_.counts.face_count,
                first_face + request_.face_count
            );
            if (first_face < face_end) {
                face_ranges.push_back({first_face, face_end - first_face, 0});
            }
        }
        if (request_.vertex_count > 0) {
            const int first_vertex = std::max(0, request_.first_vertex);
            const int vertex_end = std::min(
                request_.counts.vertex_count,
                first_vertex + request_.vertex_count
            );
            if (first_vertex < vertex_end) {
                vertex_ranges.push_back({first_vertex, vertex_end - first_vertex, 0});
            }
        }
        upload_active_face_ranges(face_ranges);
        upload_active_vertex_ranges(vertex_ranges);
        active_face_count_ = range_count(face_ranges);
        active_vertex_count_ = vertex_range_count(vertex_ranges);
    }

    static void append_active_face_range(
        std::vector<AccumulatedFaceRange>& ranges,
        const AccumulatedFrameChunk& chunk,
        int face_count
    )
    {
        const int first_face = std::max(0, chunk.first_face);
        const int face_end = std::min(
            face_count,
            first_face + std::max(0, chunk.face_count)
        );
        if (first_face >= face_end) {
            return;
        }
        if (!ranges.empty()) {
            AccumulatedFaceRange& previous = ranges.back();
            const int previous_end = previous.first_face + previous.face_count;
            if (previous_end == first_face) {
                previous.face_count += face_end - first_face;
                return;
            }
        }
        const int first_active = ranges.empty() ? 0
            : ranges.back().first_active + ranges.back().face_count;
        ranges.push_back({first_face, face_end - first_face, first_active});
    }

    static void append_active_vertex_range(
        std::vector<AccumulatedVertexRange>& ranges,
        const AccumulatedFrameChunk& chunk,
        int vertex_count
    )
    {
        const int first_vertex = std::max(0, chunk.first_vertex);
        const int vertex_end = std::min(
            vertex_count,
            first_vertex + std::max(0, chunk.vertex_count)
        );
        if (first_vertex >= vertex_end) {
            return;
        }
        if (!ranges.empty()) {
            AccumulatedVertexRange& previous = ranges.back();
            const int previous_end = previous.first_vertex + previous.vertex_count;
            if (previous_end == first_vertex) {
                previous.vertex_count += vertex_end - first_vertex;
                return;
            }
        }
        const int first_active = ranges.empty() ? 0
            : ranges.back().first_active + ranges.back().vertex_count;
        ranges.push_back({first_vertex, vertex_end - first_vertex, first_active});
    }

    static int range_count(const std::vector<AccumulatedFaceRange>& ranges)
    {
        return ranges.empty() ? 0 : ranges.back().first_active + ranges.back().face_count;
    }

    static int vertex_range_count(const std::vector<AccumulatedVertexRange>& ranges)
    {
        return ranges.empty() ? 0 : ranges.back().first_active + ranges.back().vertex_count;
    }

    void upload_active_face_ranges(const std::vector<AccumulatedFaceRange>& ranges)
    {
        auto& device_ranges = workspace().active_face_ranges;
        if (ranges.empty()) {
            device_ranges.clear();
            return;
        }
        resize_with_headroom(device_ranges, ranges.size());
        copy_host_to_device_sync(
            thrust::raw_pointer_cast(device_ranges.data()),
            ranges.data(),
            ranges.size(),
            request_.stream
        );
    }

    void upload_active_vertex_ranges(const std::vector<AccumulatedVertexRange>& ranges)
    {
        auto& device_ranges = workspace().active_vertex_ranges;
        if (ranges.empty()) {
            device_ranges.clear();
            return;
        }
        resize_with_headroom(device_ranges, ranges.size());
        copy_host_to_device_sync(
            thrust::raw_pointer_cast(device_ranges.data()),
            ranges.data(),
            ranges.size(),
            request_.stream
        );
    }

    void clear_depth()
    {
        timed_cuda_stage_void(request_.timing, "accum.clear_depth", [&] {
            resize_with_headroom(
                workspace().accumulated_depth,
                static_cast<std::size_t>(geometry_.height * geometry_.width)
            );
            thrust::fill(
                thrust::cuda::par.on(request_.stream),
                workspace().accumulated_depth.begin(),
                workspace().accumulated_depth.end(),
                std::numeric_limits<float>::infinity()
            );
        });
    }

    ProjectionBuffers projection_buffers() const
    {
        return {
            thrust::raw_pointer_cast(workspace().accumulated_vertices_global.data()),
            {
                thrust::raw_pointer_cast(workspace().projected_rows.data()),
                thrust::raw_pointer_cast(workspace().projected_cols.data()),
                thrust::raw_pointer_cast(workspace().projected_ranges.data())
            },
            workspace().active_vertex_ranges.empty()
                ? nullptr
                : thrust::raw_pointer_cast(workspace().active_vertex_ranges.data()),
            static_cast<int>(workspace().active_vertex_ranges.size())
        };
    }

    void resize_projection_buffers()
    {
        resize_with_headroom(
            workspace().projected_rows,
            static_cast<std::size_t>(request_.counts.vertex_count)
        );
        resize_with_headroom(
            workspace().projected_cols,
            static_cast<std::size_t>(request_.counts.vertex_count)
        );
        resize_with_headroom(
            workspace().projected_ranges,
            static_cast<std::size_t>(request_.counts.vertex_count)
        );
    }

    void project_vertices()
    {
        if (active_vertex_count_ <= 0) {
            return;
        }
        timed_cuda_stage_void(request_.timing, "accum.transform_and_project_vertices", [&] {
            resize_projection_buffers();
            launch_fast_projection();
        });
    }

    void launch_fast_projection()
    {
        transform_and_project_vertices_fast_kernel
            <<<grid_for(active_vertex_count_), kBlockSize, 0, request_.stream>>>(
                projection_buffers(),
                {
                    global_to_local_fast_,
                    geometry_,
                    static_cast<float>(config().vertical_fov.min_rad),
                    static_cast<float>(config().vertical_fov.max_rad - config().vertical_fov.min_rad),
                    active_vertex_count_
                }
            );
    }

    ProjectedMeshView projected_mesh() const
    {
        return {
            thrust::raw_pointer_cast(workspace().accumulated_faces_global.data()),
            {
                thrust::raw_pointer_cast(workspace().projected_rows.data()),
                thrust::raw_pointer_cast(workspace().projected_cols.data()),
                thrust::raw_pointer_cast(workspace().projected_ranges.data())
            },
            workspace().active_face_ranges.empty()
                ? nullptr
                : thrust::raw_pointer_cast(workspace().active_face_ranges.data()),
            static_cast<int>(workspace().active_face_ranges.size()),
            active_face_count_
        };
    }

    DepthTarget depth_target() const
    {
        return {
            thrust::raw_pointer_cast(workspace().accumulated_depth.data()),
            geometry_.width
        };
    }

    FaceRasterView face_raster() const
    {
        return {projected_mesh(), depth_target(), geometry_};
    }

    void rasterize_split_by_size()
    {
        if (active_face_count_ <= 0) {
            return;
        }
        mark_large_faces();
        scan_large_faces();
        if (large_face_count_ > 0) {
            compact_large_faces();
        }
        rasterize_small_faces();
        if (large_face_count_ > 0) {
            rasterize_large_faces();
        }
    }

    void mark_large_faces()
    {
        resize_with_headroom(workspace().flags, static_cast<std::size_t>(active_face_count_));
        resize_with_headroom(workspace().offsets, static_cast<std::size_t>(active_face_count_));
        timed_cuda_stage_void(request_.timing, "accum.mark_large_faces", [&] {
            mark_large_raster_faces_kernel
                <<<grid_for(active_face_count_), kBlockSize, 0, request_.stream>>>(
                    projected_mesh(),
                    thrust::raw_pointer_cast(workspace().flags.data()),
                    geometry_,
                    kLargeRasterFacePixelThreshold
                );
            CUDA_CHECK(cudaGetLastError());
        });
    }

    void scan_large_faces()
    {
        timed_cuda_stage_void(request_.timing, "accum.large_face_prefix_sum", [&] {
            large_face_count_ = scanned_count(workspace().flags, workspace().offsets, request_.stream);
            resize_with_headroom(workspace().large_face_indices, static_cast<std::size_t>(large_face_count_));
        });
    }

    void compact_large_faces()
    {
        timed_cuda_stage_void(request_.timing, "accum.compact_large_faces", [&] {
            compact_large_face_indices_kernel
                <<<grid_for(active_face_count_), kBlockSize, 0, request_.stream>>>(
                    thrust::raw_pointer_cast(workspace().flags.data()),
                    thrust::raw_pointer_cast(workspace().offsets.data()),
                    thrust::raw_pointer_cast(workspace().large_face_indices.data()),
                    active_face_count_
                );
        });
    }

    void rasterize_small_faces()
    {
        timed_cuda_stage_void(request_.timing, "accum.rasterize_small_faces", [&] {
            rasterize_accumulated_faces_fast_kernel
                <<<grid_for(active_face_count_), kBlockSize, 0, request_.stream>>>(face_raster());
        });
    }

    void rasterize_large_faces()
    {
        timed_cuda_stage_void(request_.timing, "accum.rasterize_large_faces", [&] {
            const LargeFaceView large_faces{
                face_raster(),
                thrust::raw_pointer_cast(workspace().large_face_indices.data()),
                large_face_count_
            };
            rasterize_large_accumulated_faces_fast_kernel
                <<<large_face_count_, kBlockSize, 0, request_.stream>>>(large_faces);
        });
    }

    AccumulatedDepthRequest request_;
    RiGeometry geometry_;
    DeviceMat4f global_to_local_fast_;
    int large_face_count_ = 0;
    int active_face_count_ = 0;
    int active_vertex_count_ = 0;
};

}  // namespace

void prepare_accumulated_depth_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    AccumulatedDepthOptions options
)
{
    AccumulatedDepthPass({
        &execution.workspace,
        options.counts,
        &options.pose,
        options.frame_chunks,
        &config,
        execution.timing,
        execution.stream,
    }).run();
}

void append_accumulated_depth_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    AccumulatedDepthAppendOptions options
)
{
    AccumulatedDepthRequest request{
        &execution.workspace,
        options.counts,
        &options.pose,
        nullptr,
        &config,
        execution.timing,
        execution.stream,
        true,
        options.first_vertex,
        options.vertex_count,
        options.first_face,
        options.face_count,
    };
    AccumulatedDepthPass(request).run();
}

}  // namespace ri_meshing::cuda_detail
