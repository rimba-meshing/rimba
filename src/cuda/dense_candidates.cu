#include "dense_meshing_internal.cuh"

#include "ri_meshing/face_orientation.hpp"
#include "cuda_utils.hpp"

#include <type_traits>

namespace ri_meshing::cuda_detail {
namespace {

constexpr unsigned char kQuadFaceSlotA = 1U;
constexpr unsigned char kQuadFaceSlotB = 2U;
constexpr unsigned char kQuadFaceAlternate = 4U;

struct RiDimensions {
    int height;
    int width;
};

template <typename Scalar>
struct ImagePoint {
    Scalar row;
    Scalar col;
};

template <typename Scalar>
struct ImageTriangle {
    ImagePoint<Scalar> p0;
    ImagePoint<Scalar> p1;
    ImagePoint<Scalar> p2;
};

template <typename Scalar>
struct BarycentricWeights {
    Scalar w0;
    Scalar w1;
    Scalar w2;
};

struct TrianglePoints {
    Vec3f a;
    Vec3f b;
    Vec3f c;
};

template <typename Scalar>
struct TriangleThresholds {
    Scalar max_edge_sq;
    Scalar min_area;
    Scalar min_quality;
};

struct TriangleValidityParams {
    TriangleThresholds<float> thresholds;
};

struct OverlapVertexView {
    const int* rows;
    const int* cols;
    const float* ranges;
};

struct OverlapDepthView {
    AccumulatedDepthView depth;
    RiDimensions dimensions;
};

template <typename Scalar>
struct OverlapThresholds {
    Scalar depth_abs_tol;
    Scalar depth_rel_tol;
    int min_pixels;
};

struct OverlapParams {
    OverlapDepthView accumulated;
    OverlapThresholds<float> thresholds;
};

struct QuadCandidateInput {
    const int* vim;
    const Vec3f* vertices;
    OverlapVertexView overlap_vertices;
};

struct QuadCandidateParams {
    RiDimensions dimensions;
    TriangleValidityParams validity;
    OverlapParams overlap;
    bool has_accumulated_depth;
};

struct FaceRecordOutput {
    FaceRecord* records;
    int* count;
    int* overflow;
    int capacity;
};


struct FaceEmissionInput {
    const int* vim;
    const Vec3f* vertices;
    const unsigned char* quad_face_flags;
    const int* keep_flags;
};

struct QuadFaceInput {
    const int* vim;
    const Vec3f* vertices;
    const unsigned char* quad_face_flags;
};

struct FaceEmissionParams {
    FaceRecordOutput output;
    RiDimensions dimensions;
    bool use_keep_flags;
};

static_assert(std::is_trivially_copyable_v<RiDimensions>);
static_assert(std::is_trivially_copyable_v<ImageTriangle<float>>);
static_assert(std::is_trivially_copyable_v<BarycentricWeights<float>>);
static_assert(std::is_trivially_copyable_v<TrianglePoints>);
static_assert(std::is_trivially_copyable_v<TriangleValidityParams>);
static_assert(std::is_trivially_copyable_v<OverlapVertexView>);
static_assert(std::is_trivially_copyable_v<OverlapDepthView>);
static_assert(std::is_trivially_copyable_v<OverlapParams>);
static_assert(std::is_trivially_copyable_v<QuadCandidateInput>);
static_assert(std::is_trivially_copyable_v<QuadCandidateParams>);
static_assert(std::is_trivially_copyable_v<FaceRecordOutput>);
static_assert(std::is_trivially_copyable_v<FaceEmissionInput>);
static_assert(std::is_trivially_copyable_v<QuadFaceInput>);
static_assert(std::is_trivially_copyable_v<FaceEmissionParams>);

TriangleValidityParams make_triangle_validity_params(const SparseMeshingConfig& config)
{
    const float max_edge_sq = static_cast<float>(config.max_triangle_dist * config.max_triangle_dist);
    return {
        {
            max_edge_sq,
            static_cast<float>(config.min_triangle_area),
            static_cast<float>(config.min_triangle_quality),
        },
    };
}

OverlapParams make_overlap_params(
    const SparseMeshingConfig& config,
    AccumulatedDepthView depth,
    RiDimensions dimensions
)
{
    return {
        {depth, dimensions},
        {
            static_cast<float>(config.overlap_depth_abs_tol),
            static_cast<float>(config.overlap_depth_rel_tol),
            config.overlap_min_pixels,
        },
    };
}

QuadCandidateInput make_quad_candidate_input(const CudaRiFrame& frame, CudaWorkspace& workspace)
{
    return {
        thrust::raw_pointer_cast(frame.filled_vim->data()),
        thrust::raw_pointer_cast(frame.vertices->data()),
        {
            thrust::raw_pointer_cast(workspace.current_vertex_rows.data()),
            thrust::raw_pointer_cast(workspace.current_vertex_cols.data()),
            thrust::raw_pointer_cast(workspace.current_vertex_ranges.data()),
        },
    };
}

FaceEmissionInput make_face_emission_input(
    const CudaRiFrame& frame,
    CudaWorkspace& workspace,
    bool use_keep_flags
)
{
    return {
        thrust::raw_pointer_cast(frame.filled_vim->data()),
        thrust::raw_pointer_cast(frame.vertices->data()),
        thrust::raw_pointer_cast(workspace.quad_face_flags.data()),
        use_keep_flags ? thrust::raw_pointer_cast(workspace.flags.data()) : nullptr,
    };
}

FaceEmissionParams make_face_emission_params(
    CudaWorkspace& workspace,
    int record_capacity,
    RiDimensions dimensions,
    bool use_keep_flags
)
{
    return {
        {
            thrust::raw_pointer_cast(workspace.face_records.data()),
            thrust::raw_pointer_cast(workspace.face_record_count.data()),
            thrust::raw_pointer_cast(workspace.face_overflow.data()),
            record_capacity,
        },
        dimensions,
        use_keep_flags,
    };
}

__device__ float device_range_f(Vec3f point)
{
    return sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
}

__device__ bool triangle_valid_f(
    TrianglePoints points,
    TriangleThresholds<float> thresholds
)
{
    const Vec3f a = points.a;
    const Vec3f b = points.b;
    const Vec3f c = points.c;
    const Vec3f ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3f ac{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3f bc{c.x - b.x, c.y - b.y, c.z - b.z};
    const float ab_sq = ab.x * ab.x + ab.y * ab.y + ab.z * ab.z;
    const float ac_sq = ac.x * ac.x + ac.y * ac.y + ac.z * ac.z;
    const float bc_sq = bc.x * bc.x + bc.y * bc.y + bc.z * bc.z;
    const float longest = fmaxf(ab_sq, fmaxf(ac_sq, bc_sq));
    if (longest <= 0.0F || longest > thresholds.max_edge_sq) {
        return false;
    }
    const Vec3f cross{
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x,
    };
    const float twice_area = device_range_f(cross);
    return 0.5F * twice_area >= thresholds.min_area && twice_area / longest >= thresholds.min_quality;
}

__device__ float accumulated_depth_at(AccumulatedDepthView depth, RiDimensions dimensions, int row, int col)
{
    if (depth.values == nullptr) {
        return __int_as_float(0x7f800000);
    }
    const int source_row = depth.bottom_up_normalized ? dimensions.height - 1 - row : row;
    const float value = depth.values[source_row * dimensions.width + col];
    if (!depth.bottom_up_normalized) {
        return value;
    }
    return value >= 0.999999F ? __int_as_float(0x7f800000) : value * 4096.0F;
}

__device__ bool compute_barycentric_f(
    ImageTriangle<float> triangle,
    ImagePoint<float> point,
    BarycentricWeights<float>* weights
)
{
    const float r0 = triangle.p0.row;
    const float c0 = triangle.p0.col;
    const float r1 = triangle.p1.row;
    const float c1 = triangle.p1.col;
    const float r2 = triangle.p2.row;
    const float c2 = triangle.p2.col;
    const float rp = point.row;
    const float cp = point.col;
    const float denom = (c1 - c2) * (r0 - r2) + (r2 - r1) * (c0 - c2);
    if (denom == 0.0F) {
        return false;
    }
    const float w0 = ((c1 - c2) * (rp - r2) + (r2 - r1) * (cp - c2)) / denom;
    const float w1 = ((c2 - c0) * (rp - r2) + (r0 - r2) * (cp - c2)) / denom;
    const float w2 = 1.0F - w0 - w1;
    if (w0 < 0.0F || w1 < 0.0F || w2 < 0.0F) {
        return false;
    }
    weights->w0 = w0;
    weights->w1 = w1;
    weights->w2 = w2;
    return true;
}

__device__ bool face_overlaps_accumulated_depth_f(
    Triangle face,
    OverlapVertexView vertices,
    OverlapDepthView accumulated,
    OverlapThresholds<float> thresholds
)
{
    const int height = accumulated.dimensions.height;
    const int width = accumulated.dimensions.width;
    if (vertices.rows[face.v0] < 0 || vertices.rows[face.v1] < 0 || vertices.rows[face.v2] < 0) {
        return false;
    }
    float r0 = static_cast<float>(vertices.rows[face.v0]);
    float c0 = static_cast<float>(vertices.cols[face.v0]);
    float range0 = vertices.ranges[face.v0];
    float r1 = static_cast<float>(vertices.rows[face.v1]);
    float c1 = static_cast<float>(vertices.cols[face.v1]);
    float range1 = vertices.ranges[face.v1];
    float r2 = static_cast<float>(vertices.rows[face.v2]);
    float c2 = static_cast<float>(vertices.cols[face.v2]);
    float range2 = vertices.ranges[face.v2];
    const float width_f = static_cast<float>(width);
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
    const int row_max = min(height - 1, static_cast<int>(ceilf(fmaxf(r0, fmaxf(r1, r2)))));
    const int col_min = static_cast<int>(floorf(fminf(c0, fminf(c1, c2))));
    const int col_max = static_cast<int>(ceilf(fmaxf(c0, fmaxf(c1, c2))));
    int conflicts = 0;
    for (int row = row_min; row <= row_max; ++row) {
        for (int col = col_min; col <= col_max; ++col) {
            int wrapped_col = col % width;
            if (wrapped_col < 0) {
                wrapped_col += width;
            }
            const float old_range = accumulated_depth_at(accumulated.depth, accumulated.dimensions, row, wrapped_col);
            if (!isfinite(old_range)) {
                continue;
            }
            BarycentricWeights<float> weights{0.0F, 0.0F, 0.0F};
            if (!compute_barycentric_f(
                    ImageTriangle<float>{{r0, c0}, {r1, c1}, {r2, c2}},
                    ImagePoint<float>{static_cast<float>(row), static_cast<float>(col)},
                    &weights
                )) {
                continue;
            }
            const float new_range = weights.w0 * range0 + weights.w1 * range1 + weights.w2 * range2;
            const float tolerance = fmaxf(thresholds.depth_abs_tol, thresholds.depth_rel_tol * old_range);
            if (fabsf(new_range - old_range) <= tolerance) {
                ++conflicts;
                if (conflicts >= thresholds.min_pixels) {
                    return true;
                }
            }
        }
    }
    return false;
}

__device__ FaceRecord make_face_record(Triangle face)
{
    const int a = min(face.v0, min(face.v1, face.v2));
    const int c = max(face.v0, max(face.v1, face.v2));
    const int b = face.v0 + face.v1 + face.v2 - a - c;
    return FaceRecord{a, b, c, face};
}

__device__ bool candidate_triangle_valid(
    Triangle face,
    const Vec3f* vertices,
    TriangleValidityParams params
)
{
    if (face.v0 == face.v1 || face.v1 == face.v2 || face.v0 == face.v2) {
        return false;
    }
    const TrianglePoints points{vertices[face.v0], vertices[face.v1], vertices[face.v2]};
    return triangle_valid_f(points, params.thresholds);
}

__device__ bool candidate_face_overlaps_accumulated_depth(
    Triangle face,
    OverlapVertexView vertices,
    OverlapParams params
)
{
    return face_overlaps_accumulated_depth_f(face, vertices, params.accumulated, params.thresholds);
}

__device__ void emit_record(Triangle face, FaceRecordOutput output)
{
    const int out = atomicAdd(output.count, 1);
    if (out < output.capacity) {
        output.records[out] = make_face_record(face);
    } else {
        *output.overflow = 1;
    }
}

__global__ void mark_quad_face_flags_kernel(
    QuadCandidateInput input,
    unsigned char* quad_face_flags,
    QuadCandidateParams params
)
{
    const int height = params.dimensions.height;
    const int width = params.dimensions.width;
    const int quad = blockIdx.x * blockDim.x + threadIdx.x;
    const int quad_count = (height - 1) * width;
    if (quad >= quad_count) {
        return;
    }
    quad_face_flags[quad] = 0;

    const int row = quad / width;
    const int col = quad - row * width;
    const int next_col = (col + 1) % width;
    const int tl = input.vim[row * width + col];
    const int tr = input.vim[row * width + next_col];
    const int bl = input.vim[(row + 1) * width + col];
    const int br = input.vim[(row + 1) * width + next_col];
    if (tl < 0 || tr < 0 || bl < 0 || br < 0) {
        return;
    }
    const Triangle fixed_a{tl, bl, tr};
    const Triangle fixed_b{tr, bl, br};
    const Triangle alternate_a{tl, bl, br};
    const Triangle alternate_b{tl, br, tr};
    const bool fixed_a_valid = candidate_triangle_valid(fixed_a, input.vertices, params.validity);
    const bool fixed_b_valid = candidate_triangle_valid(fixed_b, input.vertices, params.validity);
    const bool alternate_a_valid = candidate_triangle_valid(alternate_a, input.vertices, params.validity);
    const bool alternate_b_valid = candidate_triangle_valid(alternate_b, input.vertices, params.validity);

    const int fixed_valid_count = static_cast<int>(fixed_a_valid) + static_cast<int>(fixed_b_valid);
    const int alternate_valid_count = static_cast<int>(alternate_a_valid) + static_cast<int>(alternate_b_valid);
    const bool use_alternate = alternate_valid_count > fixed_valid_count;

    const Triangle face_a = use_alternate ? alternate_a : fixed_a;
    const Triangle face_b = use_alternate ? alternate_b : fixed_b;
    const bool face_a_valid = use_alternate ? alternate_a_valid : fixed_a_valid;
    const bool face_b_valid = use_alternate ? alternate_b_valid : fixed_b_valid;
    unsigned char flags = 0;

    if (params.has_accumulated_depth) {
        if (face_a_valid
            && !candidate_face_overlaps_accumulated_depth(
                face_a, input.overlap_vertices, params.overlap
            )) {
            flags |= kQuadFaceSlotA;
        }
        if (face_b_valid
            && !candidate_face_overlaps_accumulated_depth(
                face_b, input.overlap_vertices, params.overlap
            )) {
            flags |= kQuadFaceSlotB;
        }
    } else {
        if (face_a_valid) {
            flags |= kQuadFaceSlotA;
        }
        if (face_b_valid) {
            flags |= kQuadFaceSlotB;
        }
    }
    if (use_alternate) {
        flags |= kQuadFaceAlternate;
    }
    quad_face_flags[quad] = flags;
}

__device__ bool quad_slot_valid(const unsigned char* quad_face_flags, int quad, int slot)
{
    const unsigned char flags = quad_face_flags[quad];
    const unsigned char mask = slot == 0 ? kQuadFaceSlotA : kQuadFaceSlotB;
    return (flags & mask) != 0;
}

__device__ Triangle quad_slot_face(
    QuadFaceInput input,
    int quad,
    int slot,
    int width
)
{
    const int row = quad / width;
    const int col = quad - row * width;
    const int next_col = (col + 1) % width;
    const int tl = input.vim[row * width + col];
    const int tr = input.vim[row * width + next_col];
    const int bl = input.vim[(row + 1) * width + col];
    const int br = input.vim[(row + 1) * width + next_col];
    const bool alternate = (input.quad_face_flags[quad] & kQuadFaceAlternate) != 0;
    Triangle face{};
    if (alternate) {
        face = slot == 0 ? Triangle{tl, bl, br} : Triangle{tl, br, tr};
    } else {
        face = slot == 0 ? Triangle{tl, bl, tr} : Triangle{tr, bl, br};
    }
    if (sensor_orientation_dot(
            input.vertices[face.v0], input.vertices[face.v1], input.vertices[face.v2]) > 0.0) {
        const int swapped = face.v1;
        face.v1 = face.v2;
        face.v2 = swapped;
    }
    return face;
}

__global__ void emit_quad_face_records_kernel(
    FaceEmissionInput input,
    FaceEmissionParams params
)
{
    const int height = params.dimensions.height;
    const int width = params.dimensions.width;
    const int slot_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int quad_count = (height - 1) * width;
    const int slot_count = quad_count * 2;
    if (slot_idx >= slot_count) {
        return;
    }
    if (params.use_keep_flags && input.keep_flags[slot_idx] == 0) {
        return;
    }
    const int quad = slot_idx / 2;
    const int local_slot = slot_idx & 1;
    if (!quad_slot_valid(input.quad_face_flags, quad, local_slot)) {
        return;
    }
    emit_record(
        quad_slot_face(
            QuadFaceInput{input.vim, input.vertices, input.quad_face_flags},
            quad,
            local_slot,
            width
        ),
        params.output
    );
}

}  // namespace

void mark_dense_candidate_faces(DenseCandidateRequest request)
{
    constexpr int block = 256;
    const auto grid_for = [](int count) { return (count + block - 1) / block; };
    const int quad_count = (request.frame.height - 1) * request.frame.width;
    timed_cuda_stage_void(request.execution.timing, "mesh.mark_candidate_faces", [&] {
        const RiDimensions dimensions{request.frame.height, request.frame.width};
        const AccumulatedDepthView depth = request.options.accumulated_depth == nullptr
            ? AccumulatedDepthView{}
            : *request.options.accumulated_depth;
        mark_quad_face_flags_kernel<<<grid_for(quad_count), block, 0, request.execution.stream>>>(
            make_quad_candidate_input(request.frame, request.execution.workspace),
            thrust::raw_pointer_cast(request.execution.workspace.quad_face_flags.data()),
            QuadCandidateParams{
                dimensions,
                make_triangle_validity_params(request.config),
                make_overlap_params(request.config, depth, dimensions),
                request.options.accumulated_depth != nullptr,
            }
        );
        CUDA_CHECK(cudaGetLastError());
    });
}

DenseEmissionResult emit_dense_candidate_records(DenseEmissionRequest request)
{
    constexpr int block = 256;
    const auto grid_for = [](int count) { return (count + block - 1) / block; };
    const int slot_count = 2 * (request.frame.height - 1) * request.frame.width;
    DenseEmissionResult result;
    timed_cuda_stage_void(request.execution.timing, "mesh.emit_candidate_faces", [&] {
        CudaWorkspace& workspace = request.execution.workspace;
        CUDA_CHECK(cudaMemsetAsync(
            thrust::raw_pointer_cast(workspace.face_record_count.data()),
            0,
            sizeof(int),
            request.execution.stream
        ));
        CUDA_CHECK(cudaMemsetAsync(
            thrust::raw_pointer_cast(workspace.face_overflow.data()),
            0,
            sizeof(int),
            request.execution.stream
        ));
        const RiDimensions dimensions{request.frame.height, request.frame.width};
        emit_quad_face_records_kernel<<<grid_for(slot_count), block, 0, request.execution.stream>>>(
            make_face_emission_input(request.frame, workspace, request.use_keep_flags),
            make_face_emission_params(
                workspace,
                request.record_capacity,
                dimensions,
                request.use_keep_flags
            )
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(
            &result.valid_record_count,
            thrust::raw_pointer_cast(workspace.face_record_count.data()),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            request.execution.stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(request.execution.stream));
    });
    CUDA_CHECK(cudaMemcpyAsync(
        &result.overflow,
        thrust::raw_pointer_cast(request.execution.workspace.face_overflow.data()),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        request.execution.stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(request.execution.stream));
    return result;
}

}  // namespace ri_meshing::cuda_detail
