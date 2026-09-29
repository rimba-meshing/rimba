#include "cuda_workflow.hpp"

#include "cuda_utils.hpp"
#include "dense_meshing_internal.cuh"
#include "device_vector_memory.hpp"
#include "ri_meshing/face_orientation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ri_meshing::cuda_detail {
namespace {

constexpr int kMaxMacroBlockSize = 16;
constexpr int kBlock = 128;
constexpr unsigned char kCandidate2 = 1U;
constexpr unsigned char kCandidate4 = 2U;
constexpr unsigned char kCandidate8 = 4U;
constexpr unsigned char kCandidate16 = 8U;

__host__ __device__ unsigned char macro_candidate_bit(int size)
{
    if (size == 2) return kCandidate2;
    if (size == 4) return kCandidate4;
    if (size == 8) return kCandidate8;
    if (size == 16) return kCandidate16;
    return 0U;
}

struct MacroMeshView {
    int height{};
    int width{};
    const int* vim{};
    const Vec3f* vertices{};
    const Vec3f* filled_xyz{};
    const int* vertex_rows{};
    const int* vertex_cols{};
    const float* vertex_ranges{};
    const unsigned char* macro_sizes{};
    const unsigned char* fine_flags{};
    const int* fine_keep{};
    AccumulatedDepthView accumulated_depth{};
};

struct MacroMeshParams {
    MeshScalar max_error{};
    MeshScalar min_area{};
    MeshScalar min_quality{};
    MeshScalar max_edge_sq{};
    MeshScalar depth_abs_tol{};
    MeshScalar depth_rel_tol{};
    int min_depth_pixels{};
};

struct MacroPlanOutput {
    RiMacroQuad* quads{};
    int* count{};
    int capacity{};
};

struct MacroCandidateLevel {
    unsigned char* flags{};
    int cell_rows{};
    int width{};
    int size{};
    unsigned char bit{};
    unsigned char child_bit{};
};

struct MacroSelectionLevelView {
    int cell_rows{};
    int width{};
    int size{};
    unsigned char bit{};
    const unsigned char* candidates{};
    unsigned char* selected{};
};

struct MacroTilePlanView {
    int cell_rows{};
    int width{};
    const unsigned char* selected{};
};

struct MacroBalanceView {
    int cell_rows{};
    int width{};
    const unsigned char* input{};
    unsigned char* output{};
};

struct MacroFaceOutput {
    FaceRecord* records{};
    int* count{};
    int* overflow{};
    int capacity{};
};

struct MacroPatch {
    int row{};
    int col{};
    int row_size{};
    int col_size{};
};

struct MacroEdgeSplits {
    unsigned long long top{};
    unsigned long long right{};
    unsigned long long bottom{};
    unsigned long long left{};
};

struct MacroBoundaryPoint {
    int vertex{};
    int row{};
    int col{};
};

struct ImageTriangle {
    MeshScalar r0{};
    MeshScalar c0{};
    MeshScalar r1{};
    MeshScalar c1{};
    MeshScalar r2{};
    MeshScalar c2{};
};

struct ImagePoint {
    MeshScalar row{};
    MeshScalar col{};
};

__device__ int wrap_col(int col, int width)
{
    return col < 0 ? col + width : (col >= width ? col - width : col);
}

__device__ int pixel_id(int row, int col, int width)
{
    return row * width + wrap_col(col, width);
}

__device__ int vertex_at(const MacroMeshView& view, int row, int col)
{
    return view.vim[pixel_id(row, col, view.width)];
}

__device__ Vec3f vertex_point(const MacroMeshView& view, int id)
{
    return view.vertices[id];
}

__device__ Triangle orient_face(Triangle face, const MacroMeshView& view)
{
    if (sensor_orientation_dot(
            vertex_point(view, face.v0), vertex_point(view, face.v1), vertex_point(view, face.v2)) > static_cast<MeshScalar>(0)) {
        const int swapped = face.v1;
        face.v1 = face.v2;
        face.v2 = swapped;
    }
    return face;
}

__device__ bool triangle_has_quality(
    Triangle face,
    const MacroMeshView& view,
    const MacroMeshParams& params,
    bool enforce_max_edge
)
{
    if (face.v0 == face.v1 || face.v1 == face.v2 || face.v0 == face.v2) {
        return false;
    }
    const Vec3f a = vertex_point(view, face.v0);
    const Vec3f b = vertex_point(view, face.v1);
    const Vec3f c = vertex_point(view, face.v2);
    const Vec3f ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3f ac{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3f bc{c.x - b.x, c.y - b.y, c.z - b.z};
    const MeshScalar ab_sq = ab.x * ab.x + ab.y * ab.y + ab.z * ab.z;
    const MeshScalar ac_sq = ac.x * ac.x + ac.y * ac.y + ac.z * ac.z;
    const MeshScalar bc_sq = bc.x * bc.x + bc.y * bc.y + bc.z * bc.z;
    const MeshScalar longest = max(ab_sq, max(ac_sq, bc_sq));
    if (longest <= static_cast<MeshScalar>(0) || (enforce_max_edge && longest > params.max_edge_sq)) {
        return false;
    }
    const Vec3f cross{
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x,
    };
    const MeshScalar twice_area = sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
    return static_cast<MeshScalar>(0.5) * twice_area >= params.min_area
        && twice_area / longest >= params.min_quality;
}

__device__ Triangle corner_macro_face(
    const MacroMeshView& view,
    MacroPatch patch,
    int index
)
{
    const int top_left = vertex_at(view, patch.row, patch.col);
    const int top_right = vertex_at(view, patch.row, patch.col + patch.col_size);
    const int bottom_left = vertex_at(view, patch.row + patch.row_size, patch.col);
    const int bottom_right = vertex_at(
        view, patch.row + patch.row_size, patch.col + patch.col_size
    );
    const Triangle face = index == 0
        ? Triangle{top_left, bottom_left, top_right}
        : Triangle{top_right, bottom_left, bottom_right};
    return orient_face(face, view);
}

__device__ int macro_tile_anchor_col(
    const MacroMeshView& view,
    int row,
    int col
)
{
    const int wrapped_col = wrap_col(col, view.width);
    const int size = view.macro_sizes[row * view.width + wrapped_col];
    return (wrapped_col / max(size, 1)) * max(size, 1);
}

__device__ int macro_tile_anchor_row(
    const MacroMeshView& view,
    int row,
    int col
)
{
    const int wrapped_row = max(0, min(row, view.height - 2));
    const int wrapped_col = wrap_col(col, view.width);
    const int size = view.macro_sizes[wrapped_row * view.width + wrapped_col];
    return (wrapped_row / max(size, 1)) * max(size, 1);
}

__device__ MacroEdgeSplits macro_edge_splits(
    const MacroMeshView& view,
    MacroPatch patch
)
{
    MacroEdgeSplits splits{};
    if (view.macro_sizes == nullptr || patch.row_size <= 1) return splits;
    const int cell_rows = view.height - 1;
    if (patch.row > 0) {
        for (int offset = 1; offset < patch.col_size; ++offset) {
            const int left = macro_tile_anchor_col(
                view, patch.row - 1, patch.col + offset - 1
            );
            const int right = macro_tile_anchor_col(
                view, patch.row - 1, patch.col + offset
            );
            if (left != right) splits.top |= 1ULL << offset;
        }
    }
    if (patch.row + patch.row_size < cell_rows) {
        for (int offset = 1; offset < patch.col_size; ++offset) {
            const int bottom_left = macro_tile_anchor_col(
                view, patch.row + patch.row_size, patch.col + offset - 1
            );
            const int bottom_right = macro_tile_anchor_col(
                view, patch.row + patch.row_size, patch.col + offset
            );
            if (bottom_left != bottom_right) splits.bottom |= 1ULL << offset;
        }
    }
    for (int offset = 1; offset < patch.row_size; ++offset) {
        const int left = macro_tile_anchor_row(
            view, patch.row + offset - 1, patch.col - 1
        );
        const int right = macro_tile_anchor_row(
            view, patch.row + offset, patch.col - 1
        );
        if (left != right) splits.left |= 1ULL << offset;
        const int right_col = wrap_col(patch.col + patch.col_size, view.width);
        const int top = macro_tile_anchor_row(
            view, patch.row + offset - 1, right_col
        );
        const int bottom = macro_tile_anchor_row(
            view, patch.row + offset, right_col
        );
        if (top != bottom) splits.right |= 1ULL << offset;
    }
    return splits;
}

__device__ int macro_boundary_vertices(
    const MacroMeshView& view,
    MacroPatch patch,
    MacroEdgeSplits splits,
    MacroBoundaryPoint* boundary
)
{
    int count = 0;
    boundary[count++] = MacroBoundaryPoint{
        vertex_at(view, patch.row, patch.col), patch.row, patch.col
    };
    for (int dc = 1; dc < patch.col_size; ++dc) {
        if ((splits.top & (1ULL << dc)) != 0ULL) {
            boundary[count++] = MacroBoundaryPoint{
                vertex_at(view, patch.row, patch.col + dc), patch.row, patch.col + dc
            };
        }
    }
    boundary[count++] = MacroBoundaryPoint{
        vertex_at(view, patch.row, patch.col + patch.col_size),
        patch.row,
        patch.col + patch.col_size,
    };
    for (int dr = 1; dr < patch.row_size; ++dr) {
        if ((splits.right & (1ULL << dr)) != 0ULL) {
            boundary[count++] = MacroBoundaryPoint{
                vertex_at(view, patch.row + dr, patch.col + patch.col_size),
                patch.row + dr,
                patch.col + patch.col_size,
            };
        }
    }
    boundary[count++] = MacroBoundaryPoint{
        vertex_at(view, patch.row + patch.row_size, patch.col + patch.col_size),
        patch.row + patch.row_size,
        patch.col + patch.col_size,
    };
    for (int dc = patch.col_size - 1; dc > 0; --dc) {
        if ((splits.bottom & (1ULL << dc)) != 0ULL) {
            boundary[count++] = MacroBoundaryPoint{
                vertex_at(view, patch.row + patch.row_size, patch.col + dc),
                patch.row + patch.row_size,
                patch.col + dc,
            };
        }
    }
    boundary[count++] = MacroBoundaryPoint{
        vertex_at(view, patch.row + patch.row_size, patch.col),
        patch.row + patch.row_size,
        patch.col,
    };
    for (int dr = patch.row_size - 1; dr > 0; --dr) {
        if ((splits.left & (1ULL << dr)) != 0ULL) {
            boundary[count++] = MacroBoundaryPoint{
                vertex_at(view, patch.row + dr, patch.col),
                patch.row + dr,
                patch.col,
            };
        }
    }
    return count;
}

__device__ MeshScalar boundary_cross(
    const MacroBoundaryPoint& a,
    const MacroBoundaryPoint& b,
    const MacroBoundaryPoint& c
)
{
    return static_cast<MeshScalar>(b.col - a.col) * (c.row - a.row)
        - static_cast<MeshScalar>(b.row - a.row) * (c.col - a.col);
}

__device__ bool boundary_point_in_triangle(
    const MacroBoundaryPoint& point,
    const MacroBoundaryPoint& a,
    const MacroBoundaryPoint& b,
    const MacroBoundaryPoint& c,
    MeshScalar orientation
)
{
    const MeshScalar ab = boundary_cross(a, b, point) * orientation;
    const MeshScalar bc = boundary_cross(b, c, point) * orientation;
    const MeshScalar ca = boundary_cross(c, a, point) * orientation;
    return ab >= static_cast<MeshScalar>(0) && bc >= static_cast<MeshScalar>(0)
        && ca >= static_cast<MeshScalar>(0);
}

__device__ bool is_boundary_ear(
    const MacroBoundaryPoint* boundary,
    const int* active,
    int remaining,
    int position,
    MeshScalar orientation
)
{
    const int previous = active[(position + remaining - 1) % remaining];
    const int current = active[position];
    const int next = active[(position + 1) % remaining];
    if (boundary_cross(boundary[previous], boundary[current], boundary[next]) * orientation
        <= static_cast<MeshScalar>(0)) {
        return false;
    }
    for (int index = 0; index < remaining; ++index) {
        const int candidate = active[index];
        if (candidate == previous || candidate == current || candidate == next) continue;
        if (boundary_point_in_triangle(
                boundary[candidate], boundary[previous], boundary[current], boundary[next], orientation
            )) {
            return false;
        }
    }
    return true;
}

__device__ MeshScalar vertex_distance_sq(
    const MacroMeshView& view,
    const MacroBoundaryPoint& a,
    const MacroBoundaryPoint& b
)
{
    const Vec3f pa = vertex_point(view, a.vertex);
    const Vec3f pb = vertex_point(view, b.vertex);
    const MeshScalar dx = pa.x - pb.x;
    const MeshScalar dy = pa.y - pb.y;
    const MeshScalar dz = pa.z - pb.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ void emit_face_record(
    const MacroFaceOutput& output,
    Triangle face
)
{
    const int out = atomicAdd(output.count, 1);
    if (out < output.capacity) {
        const int low = min(face.v0, min(face.v1, face.v2));
        const int high = max(face.v0, max(face.v1, face.v2));
        output.records[out] = FaceRecord{
            low, face.v0 + face.v1 + face.v2 - low - high, high, face
        };
    } else {
        *output.overflow = 1;
    }
}

__device__ bool barycentric(
    ImageTriangle triangle,
    ImagePoint point,
    MeshScalar* w0,
    MeshScalar* w1,
    MeshScalar* w2
)
{
    const MeshScalar denom = (triangle.c1 - triangle.c2) * (triangle.r0 - triangle.r2)
        + (triangle.r2 - triangle.r1) * (triangle.c0 - triangle.c2);
    if (denom == static_cast<MeshScalar>(0)) {
        return false;
    }
    *w0 = ((triangle.c1 - triangle.c2) * (point.row - triangle.r2)
        + (triangle.r2 - triangle.r1) * (point.col - triangle.c2)) / denom;
    *w1 = ((triangle.c2 - triangle.c0) * (point.row - triangle.r2)
        + (triangle.r0 - triangle.r2) * (point.col - triangle.c2)) / denom;
    *w2 = static_cast<MeshScalar>(1) - *w0 - *w1;
    return *w0 >= static_cast<MeshScalar>(0) && *w1 >= static_cast<MeshScalar>(0)
        && *w2 >= static_cast<MeshScalar>(0);
}

__device__ bool macro_face_overlaps_depth(
    Triangle face,
    const MacroMeshView& view,
    const MacroMeshParams& params
)
{
    if (view.accumulated_depth.values == nullptr) {
        return false;
    }
    const int a = face.v0;
    const int b = face.v1;
    const int c = face.v2;
    if (view.vertex_rows[a] < 0 || view.vertex_rows[b] < 0 || view.vertex_rows[c] < 0) {
        return false;
    }
    MeshScalar r0 = static_cast<MeshScalar>(view.vertex_rows[a]);
    MeshScalar c0 = static_cast<MeshScalar>(view.vertex_cols[a]);
    MeshScalar r1 = static_cast<MeshScalar>(view.vertex_rows[b]);
    MeshScalar c1 = static_cast<MeshScalar>(view.vertex_cols[b]);
    MeshScalar r2 = static_cast<MeshScalar>(view.vertex_rows[c]);
    MeshScalar c2 = static_cast<MeshScalar>(view.vertex_cols[c]);
    if (max(c0, max(c1, c2)) - min(c0, min(c1, c2))
        > static_cast<MeshScalar>(view.width) / static_cast<MeshScalar>(2)) {
        if (c0 < static_cast<MeshScalar>(view.width) / static_cast<MeshScalar>(2)) c0 += view.width;
        if (c1 < static_cast<MeshScalar>(view.width) / static_cast<MeshScalar>(2)) c1 += view.width;
        if (c2 < static_cast<MeshScalar>(view.width) / static_cast<MeshScalar>(2)) c2 += view.width;
    }
    const int row_min = max(0, static_cast<int>(floor(min(r0, min(r1, r2)))));
    const int row_max = min(view.height - 1, static_cast<int>(ceil(max(r0, max(r1, r2)))));
    const int col_min = static_cast<int>(floor(min(c0, min(c1, c2))));
    const int col_max = static_cast<int>(ceil(max(c0, max(c1, c2))));
    int conflicts = 0;
    for (int row = row_min; row <= row_max; ++row) {
        for (int col = col_min; col <= col_max; ++col) {
            int wrapped = col % view.width;
            if (wrapped < 0) wrapped += view.width;
            const int source_row = view.accumulated_depth.bottom_up_normalized
                ? view.height - 1 - row : row;
            float old_range = view.accumulated_depth.values[source_row * view.width + wrapped];
            if (view.accumulated_depth.bottom_up_normalized) {
                old_range = old_range >= 0.999999F ? __int_as_float(0x7f800000) : old_range * 4096.0F;
            }
            if (!isfinite(old_range)) continue;
            MeshScalar w0{}, w1{}, w2{};
            if (!barycentric(ImageTriangle{r0, c0, r1, c1, r2, c2},
                    ImagePoint{static_cast<MeshScalar>(row), static_cast<MeshScalar>(col)},
                    &w0, &w1, &w2)) continue;
            const MeshScalar new_range = w0 * view.vertex_ranges[a]
                + w1 * view.vertex_ranges[b] + w2 * view.vertex_ranges[c];
            const MeshScalar tolerance = max(params.depth_abs_tol, params.depth_rel_tol * old_range);
            if (fabs(new_range - old_range) <= tolerance && ++conflicts >= params.min_depth_pixels) {
                return true;
            }
        }
    }
    return false;
}

__device__ bool all_samples_present(
    const MacroMeshView& view,
    int row,
    int col,
    int row_size,
    int col_size
)
{
    for (int dr = 0; dr <= row_size; ++dr) {
        for (int dc = 0; dc <= col_size; ++dc) {
            if (vertex_at(view, row + dr, col + dc) < 0) return false;
        }
    }
    return true;
}

__device__ bool bilinear_error_is_small(
    const MacroMeshView& view,
    MacroPatch patch,
    MeshScalar max_error
)
{
    const Vec3f p00 = vertex_point(view, vertex_at(view, patch.row, patch.col));
    const Vec3f p10 = vertex_point(view, vertex_at(view, patch.row, patch.col + patch.col_size));
    const Vec3f p01 = vertex_point(view, vertex_at(view, patch.row + patch.row_size, patch.col));
    const Vec3f p11 = vertex_point(
        view, vertex_at(view, patch.row + patch.row_size, patch.col + patch.col_size)
    );
    const MeshScalar max_error_sq = max_error * max_error;
    for (int dr = 0; dr <= patch.row_size; ++dr) {
        const MeshScalar v = static_cast<MeshScalar>(dr) / patch.row_size;
        for (int dc = 0; dc <= patch.col_size; ++dc) {
            const MeshScalar u = static_cast<MeshScalar>(dc) / patch.col_size;
            const Vec3f top{
                static_cast<float>((static_cast<MeshScalar>(1) - u) * p00.x + u * p10.x),
                static_cast<float>((static_cast<MeshScalar>(1) - u) * p00.y + u * p10.y),
                static_cast<float>((static_cast<MeshScalar>(1) - u) * p00.z + u * p10.z),
            };
            const Vec3f bottom{
                static_cast<float>((static_cast<MeshScalar>(1) - u) * p01.x + u * p11.x),
                static_cast<float>((static_cast<MeshScalar>(1) - u) * p01.y + u * p11.y),
                static_cast<float>((static_cast<MeshScalar>(1) - u) * p01.z + u * p11.z),
            };
            const Vec3f expected{
                static_cast<float>((static_cast<MeshScalar>(1) - v) * top.x + v * bottom.x),
                static_cast<float>((static_cast<MeshScalar>(1) - v) * top.y + v * bottom.y),
                static_cast<float>((static_cast<MeshScalar>(1) - v) * top.z + v * bottom.z),
            };
            const Vec3f actual = view.filled_xyz[
                pixel_id(patch.row + dr, patch.col + dc, view.width)
            ];
            const MeshScalar dx = actual.x - expected.x;
            const MeshScalar dy = actual.y - expected.y;
            const MeshScalar dz = actual.z - expected.z;
            if (dx * dx + dy * dy + dz * dz > max_error_sq) return false;
        }
    }
    return true;
}

__device__ bool block_can_be_corner_macro(
    const MacroMeshView& view,
    const MacroMeshParams& params,
    MacroPatch patch,
    bool children_cover_presence
)
{
    if (patch.row_size < 2 || patch.col_size < 2) {
        return false;
    }
    if ((!children_cover_presence
            && !all_samples_present(view, patch.row, patch.col, patch.row_size, patch.col_size))
        || !bilinear_error_is_small(view, patch, params.max_error)) {
        return false;
    }
    for (int index = 0; index < 2; ++index) {
        const Triangle face = corner_macro_face(view, patch, index);
        if (!triangle_has_quality(face, view, params, false)
            || macro_face_overlaps_depth(face, view, params)) {
            return false;
        }
    }
    return true;
}

__device__ bool level_children_are_candidates(
    const MacroCandidateLevel& level,
    int row,
    int col
)
{
    if (level.child_bit == 0U) return true;
    const int child_size = level.size / 2;
    return (level.flags[pixel_id(row, col, level.width)] & level.child_bit) != 0U
        && (level.flags[pixel_id(row, col + child_size, level.width)] & level.child_bit) != 0U
        && (level.flags[pixel_id(row + child_size, col, level.width)] & level.child_bit) != 0U
        && (level.flags[pixel_id(row + child_size, col + child_size, level.width)] & level.child_bit) != 0U;
}

__global__ void evaluate_macro_level_kernel(
    MacroMeshView view,
    MacroMeshParams params,
    MacroCandidateLevel level
)
{
    // This is the implicit quadtree: each launch materializes one level in
    // bottom-up order, and parent candidates depend only on four child flags.
    const int columns = (level.width + level.size - 1) / level.size;
    const int rows = (level.cell_rows + level.size - 1) / level.size;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * columns) return;

    const int row = (index / columns) * level.size;
    const int col = (index % columns) * level.size;
    if (row + level.size > level.cell_rows || col + level.size > level.width) return;
    if (!level_children_are_candidates(level, row, col)) return;
    if (block_can_be_corner_macro(
            view,
            params,
            MacroPatch{row, col, level.size, level.size},
            level.child_bit != 0U)) {
        level.flags[pixel_id(row, col, level.width)] |= level.bit;
    }
}

__global__ void select_macro_size_level_kernel(MacroSelectionLevelView view)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int columns = (view.width + view.size - 1) / view.size;
    const int rows = (view.cell_rows + view.size - 1) / view.size;
    if (index >= rows * columns) return;

    const int row = (index / columns) * view.size;
    const int col = (index % columns) * view.size;
    if (row + view.size > view.cell_rows || col + view.size > view.width
        || (view.candidates[pixel_id(row, col, view.width)] & view.bit) == 0U) {
        return;
    }

    // Candidate levels are processed from coarse to fine. Tiles at one level
    // are disjoint, so this loop can fill the selected map without atomics;
    // the selected==1 guard preserves a larger tile written by an earlier
    // level.
    for (int dr = 0; dr < view.size; ++dr) {
        for (int dc = 0; dc < view.size; ++dc) {
            const int target_row = row + dr;
            const int target_col = col + dc;
            if (target_row + view.size <= view.cell_rows
                && target_col + view.size <= view.width) {
                const int target = target_row * view.width + target_col;
                if (view.selected[target] == 1U) {
                    view.selected[target] = static_cast<unsigned char>(view.size);
                }
            }
        }
    }
}

__global__ void balance_macro_sizes_kernel(MacroBalanceView view)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = view.cell_rows * view.width;
    if (index >= cell_count) return;
    const int row = index / view.width;
    const int col = index - row * view.width;
    const int size = view.input[index];
    if (size <= 1) {
        view.output[index] = 1U;
        return;
    }

    // A coarse tile is split as a whole when any part of one of its four
    // edges borders a finer tile. This is the conforming-mesh pass; the
    // larger tile splits, while the already-finer neighbor stays unchanged.
    const int anchor_row = (row / size) * size;
    const int anchor_col = (col / size) * size;
    bool split = false;
    for (int offset = 0; offset < size && !split; ++offset) {
        const int left_col = wrap_col(anchor_col - 1, view.width);
        const int right_col = wrap_col(anchor_col + size, view.width);
        if (view.input[(anchor_row + offset) * view.width + left_col] < size
            || view.input[(anchor_row + offset) * view.width + right_col] < size) {
            split = true;
        }
    }
    for (int offset = 0; offset < size && !split; ++offset) {
        if (anchor_row > 0
            && view.input[(anchor_row - 1) * view.width + wrap_col(anchor_col + offset, view.width)] < size) {
            split = true;
        }
        if (anchor_row + size < view.cell_rows
            && view.input[(anchor_row + size) * view.width + wrap_col(anchor_col + offset, view.width)] < size) {
            split = true;
        }
    }
    view.output[index] = split ? static_cast<unsigned char>(size / 2) : static_cast<unsigned char>(size);
}

__global__ void emit_macro_tile_plan_kernel(
    MacroTilePlanView view,
    MacroPlanOutput output
)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = view.cell_rows * view.width;
    if (index >= cell_count) return;
    const int row = index / view.width;
    const int col = index - row * view.width;
    const int size = view.selected[index];
    if (size <= 0 || row % size != 0 || col % size != 0
        || row + size > view.cell_rows || col + size > view.width) {
        return;
    }
    const int out = atomicAdd(output.count, 1);
    if (out < output.capacity) {
        output.quads[out] = RiMacroQuad{
            row, col, size, size, static_cast<unsigned char>(size > 1 ? 1 : 0)
        };
    }
}

__device__ void emit_unit_quad(
    const MacroMeshView& view,
    const MacroFaceOutput& output,
    int row,
    int col
)
{
    const int quad = row * view.width + wrap_col(col, view.width);
    const unsigned char flags = view.fine_flags[quad];
    for (int slot = 0; slot < 2; ++slot) {
        if ((flags & (slot == 0 ? 1U : 2U)) == 0) continue;
        const int slot_index = 2 * quad + slot;
        if (view.fine_keep != nullptr && view.fine_keep[slot_index] == 0) continue;
        const int next_col = wrap_col(col + 1, view.width);
        const int tl = view.vim[row * view.width + wrap_col(col, view.width)];
        const int tr = view.vim[row * view.width + next_col];
        const int bl = view.vim[(row + 1) * view.width + wrap_col(col, view.width)];
        const int br = view.vim[(row + 1) * view.width + next_col];
        const bool alternate = (flags & 4U) != 0;
        Triangle face = alternate
            ? (slot == 0 ? Triangle{tl, bl, br} : Triangle{tl, br, tr})
            : (slot == 0 ? Triangle{tl, bl, tr} : Triangle{tr, bl, br});
        face = orient_face(face, view);
        const int out = atomicAdd(output.count, 1);
        if (out < output.capacity) output.records[out] = FaceRecord{
            min(face.v0, min(face.v1, face.v2)),
            face.v0 + face.v1 + face.v2 - min(face.v0, min(face.v1, face.v2))
                - max(face.v0, max(face.v1, face.v2)),
            max(face.v0, max(face.v1, face.v2)),
            face,
        };
        else *output.overflow = 1;
    }
}

__device__ void emit_conforming_macro_faces(
    const MacroMeshView& view,
    const MacroFaceOutput& output,
    MacroPatch patch,
    MacroEdgeSplits splits
)
{
    MacroBoundaryPoint boundary[4 * kMaxMacroBlockSize];
    const int boundary_count = macro_boundary_vertices(
        view, patch, splits, boundary
    );
    if (boundary_count < 3) return;

    int active[4 * kMaxMacroBlockSize];
    for (int index = 0; index < boundary_count; ++index) active[index] = index;
    MeshScalar polygon_area = static_cast<MeshScalar>(0);
    for (int index = 0; index < boundary_count; ++index) {
        const MacroBoundaryPoint& current = boundary[index];
        const MacroBoundaryPoint& next = boundary[(index + 1) % boundary_count];
        polygon_area += static_cast<MeshScalar>(current.col) * next.row
            - static_cast<MeshScalar>(next.col) * current.row;
    }
    if (polygon_area == static_cast<MeshScalar>(0)) return;
    const MeshScalar orientation = polygon_area > static_cast<MeshScalar>(0)
        ? static_cast<MeshScalar>(1) : static_cast<MeshScalar>(-1);
    int remaining = boundary_count;
    while (remaining > 3) {
        int best_position = -1;
        MeshScalar best_diagonal = static_cast<MeshScalar>(1.0e30);
        for (int position = 0; position < remaining; ++position) {
            if (!is_boundary_ear(
                    boundary, active, remaining, position, orientation
                )) {
                continue;
            }
            const int previous = active[(position + remaining - 1) % remaining];
            const int next = active[(position + 1) % remaining];
            const MeshScalar diagonal = vertex_distance_sq(
                view, boundary[previous], boundary[next]
            );
            if (diagonal < best_diagonal) {
                best_diagonal = diagonal;
                best_position = position;
            }
        }
        if (best_position < 0) {
            const int center = vertex_at(
                view, patch.row + patch.row_size / 2,
                patch.col + patch.col_size / 2
            );
            for (int index = 0; index < remaining; ++index) {
                const Triangle face = orient_face(
                    Triangle{
                        center,
                        boundary[active[index]].vertex,
                        boundary[active[(index + 1) % remaining]].vertex,
                    },
                    view
                );
                emit_face_record(output, face);
            }
            return;
        }
        const int previous = active[(best_position + remaining - 1) % remaining];
        const int current = active[best_position];
        const int next = active[(best_position + 1) % remaining];
        emit_face_record(
            output,
            orient_face(
                Triangle{
                    boundary[previous].vertex,
                    boundary[current].vertex,
                    boundary[next].vertex,
                },
                view
            )
        );
        for (int position = best_position; position + 1 < remaining; ++position) {
            active[position] = active[position + 1];
        }
        --remaining;
    }
    emit_face_record(
        output,
        orient_face(
            Triangle{
                boundary[active[0]].vertex,
                boundary[active[1]].vertex,
                boundary[active[2]].vertex,
            },
            view
        )
    );
}

__global__ void emit_macro_faces_kernel(
    MacroMeshView view,
    const RiMacroQuad* quads,
    const int* quad_count_device,
    int quad_capacity,
    MacroFaceOutput output
)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= *quad_count_device || index >= quad_capacity) return;
    const RiMacroQuad quad = quads[index];
    const MacroPatch patch{quad.row, quad.col, quad.row_size, quad.col_size};
    if (quad.corner_triangles == 0U) {
        emit_unit_quad(view, output, quad.row, quad.col);
        return;
    }
    const MacroEdgeSplits splits = macro_edge_splits(view, patch);
    if (splits.top != 0U || splits.right != 0U
        || splits.bottom != 0U || splits.left != 0U) {
        emit_conforming_macro_faces(view, output, patch, splits);
        return;
    }
    for (int face_index = 0; face_index < 2; ++face_index) {
        const Triangle face = corner_macro_face(
            view, MacroPatch{quad.row, quad.col, quad.row_size, quad.col_size}, face_index
        );
        const int out = atomicAdd(output.count, 1);
        if (out < output.capacity) {
            const int low = min(face.v0, min(face.v1, face.v2));
            const int high = max(face.v0, max(face.v1, face.v2));
            output.records[out] = FaceRecord{
                low, face.v0 + face.v1 + face.v2 - low - high, high, face
            };
        } else {
            *output.overflow = 1;
        }
    }
}

MacroMeshView make_macro_view(
    const CudaRiFrame& frame,
    CudaWorkspace& workspace,
    const DenseMeshOptions& options,
    bool use_fine_keep
)
{
    return MacroMeshView{
        frame.height,
        frame.width,
        thrust::raw_pointer_cast(frame.filled_vim->data()),
        thrust::raw_pointer_cast(frame.vertices->data()),
        thrust::raw_pointer_cast(workspace.filled_pixel_xyz.data()),
        thrust::raw_pointer_cast(workspace.current_vertex_rows.data()),
        thrust::raw_pointer_cast(workspace.current_vertex_cols.data()),
        thrust::raw_pointer_cast(workspace.current_vertex_ranges.data()),
        thrust::raw_pointer_cast(workspace.macro_selected_size.data()),
        thrust::raw_pointer_cast(workspace.quad_face_flags.data()),
        use_fine_keep ? thrust::raw_pointer_cast(workspace.flags.data()) : nullptr,
        options.accumulated_depth == nullptr ? AccumulatedDepthView{} : *options.accumulated_depth,
    };
}

MacroMeshParams make_macro_params(const SparseMeshingConfig& config)
{
    return {
        config.filled_pixel_macro_max_error,
        config.min_triangle_area,
        config.min_triangle_quality,
        config.max_triangle_dist * config.max_triangle_dist,
        config.overlap_depth_abs_tol,
        config.overlap_depth_rel_tol,
        config.overlap_min_pixels,
    };
}

void build_macro_quad_plan(
    const CudaRiFrame& frame,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    const DenseMeshOptions& options
)
{
    CudaWorkspace& workspace = execution.workspace;
    const int cell_rows = frame.height - 1;
    const int cell_count = cell_rows * frame.width;
    const int quad_count = cell_count;
    resize_with_headroom(workspace.macro_quads, static_cast<std::size_t>(quad_count));
    resize_with_headroom(workspace.macro_quad_count, 1);
    resize_with_headroom(workspace.macro_candidate_flags, static_cast<std::size_t>(cell_count));
    resize_with_headroom(workspace.macro_selected_size, static_cast<std::size_t>(cell_count));
    resize_with_headroom(workspace.macro_next_size, static_cast<std::size_t>(cell_count));
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(workspace.macro_candidate_flags.data()), 0,
        static_cast<std::size_t>(cell_count) * sizeof(unsigned char), execution.stream
    ));
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(workspace.macro_quad_count.data()), 0, sizeof(int), execution.stream
    ));
    const MacroMeshView view = make_macro_view(
        frame, workspace, options, config.dense_isolated_component_max_faces > 0
    );
    const MacroMeshParams params = make_macro_params(config);
    timed_cuda_stage_void(execution.timing, "mesh.plan_macro_quads", [&] {
        for (int size = 2; size <= kMaxMacroBlockSize; size *= 2) {
            const int level_count = ((frame.width + size - 1) / size)
                * ((cell_rows + size - 1) / size);
            evaluate_macro_level_kernel<<<
                (level_count + kBlock - 1) / kBlock, kBlock, 0, execution.stream
            >>>(
                view, params, MacroCandidateLevel{
                    thrust::raw_pointer_cast(workspace.macro_candidate_flags.data()),
                    cell_rows,
                    frame.width,
                    size,
                    macro_candidate_bit(size),
                    static_cast<unsigned char>(
                        size == 2 ? 0 : macro_candidate_bit(size / 2)
                    ),
                }
            );
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemsetAsync(
            thrust::raw_pointer_cast(workspace.macro_selected_size.data()),
            1,
            static_cast<std::size_t>(cell_count) * sizeof(unsigned char),
            execution.stream
        ));
        for (int size = kMaxMacroBlockSize; size >= 2; size /= 2) {
            const int level_count = ((frame.width + size - 1) / size)
                * ((cell_rows + size - 1) / size);
            select_macro_size_level_kernel<<<
                (level_count + kBlock - 1) / kBlock, kBlock, 0, execution.stream
            >>>(
                MacroSelectionLevelView{
                    cell_rows,
                    frame.width,
                    size,
                    macro_candidate_bit(size),
                    thrust::raw_pointer_cast(workspace.macro_candidate_flags.data()),
                    thrust::raw_pointer_cast(workspace.macro_selected_size.data()),
                }
            );
            CUDA_CHECK(cudaGetLastError());
        }
        for (int pass = 0, size = 2; size <= kMaxMacroBlockSize; ++pass, size *= 2) {
            balance_macro_sizes_kernel<<<(cell_count + kBlock - 1) / kBlock, kBlock, 0, execution.stream>>>(
                MacroBalanceView{
                    cell_rows, frame.width,
                    thrust::raw_pointer_cast(workspace.macro_selected_size.data()),
                    thrust::raw_pointer_cast(workspace.macro_next_size.data()),
                }
            );
            CUDA_CHECK(cudaGetLastError());
            workspace.macro_selected_size.swap(workspace.macro_next_size);
        }
        emit_macro_tile_plan_kernel<<<(cell_count + kBlock - 1) / kBlock, kBlock, 0, execution.stream>>>(
            MacroTilePlanView{
                cell_rows, frame.width,
                thrust::raw_pointer_cast(workspace.macro_selected_size.data()),
            },
            MacroPlanOutput{
                thrust::raw_pointer_cast(workspace.macro_quads.data()),
                thrust::raw_pointer_cast(workspace.macro_quad_count.data()),
                quad_count,
            }
        );
        CUDA_CHECK(cudaGetLastError());
    });
}

thrust::device_vector<Triangle>& emit_coarsened_faces(
    const CudaRiFrame& frame,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    DenseMeshOptions options
)
{
    CudaWorkspace& workspace = execution.workspace;
    const int quad_count = (frame.height - 1) * frame.width;
    const int max_face_count = 2 * quad_count;
    const int record_capacity = std::min(
        max_face_count,
        static_cast<int>(frame.vertices->size() * 4 + 1024)
    );
    resize_with_headroom(workspace.face_records, static_cast<std::size_t>(record_capacity));
    resize_with_headroom(workspace.face_record_count, 1);
    resize_with_headroom(workspace.face_overflow, 1);
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(workspace.face_record_count.data()), 0, sizeof(int), execution.stream
    ));
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(workspace.face_overflow.data()), 0, sizeof(int), execution.stream
    ));
    const MacroMeshView view = make_macro_view(
        frame, workspace, options, config.dense_isolated_component_max_faces > 0
    );
    timed_cuda_stage_void(execution.timing, "mesh.emit_macro_faces", [&] {
        emit_macro_faces_kernel<<<(quad_count + kBlock - 1) / kBlock, kBlock, 0, execution.stream>>>(
            view,
            thrust::raw_pointer_cast(workspace.macro_quads.data()),
            thrust::raw_pointer_cast(workspace.macro_quad_count.data()),
            quad_count,
            MacroFaceOutput{
                thrust::raw_pointer_cast(workspace.face_records.data()),
                thrust::raw_pointer_cast(workspace.face_record_count.data()),
                thrust::raw_pointer_cast(workspace.face_overflow.data()),
                record_capacity,
            }
        );
        CUDA_CHECK(cudaGetLastError());
    });
    int record_count = 0;
    int overflow = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &record_count,
        thrust::raw_pointer_cast(workspace.face_record_count.data()),
        sizeof(int), cudaMemcpyDeviceToHost, execution.stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        &overflow,
        thrust::raw_pointer_cast(workspace.face_overflow.data()),
        sizeof(int), cudaMemcpyDeviceToHost, execution.stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(execution.stream));
    if (overflow != 0 || record_count > record_capacity) {
        throw std::runtime_error("Coarsened dense meshing face record capacity exceeded");
    }
    if (record_count == 0) {
        workspace.faces.clear();
        return workspace.faces;
    }
    return compact_dense_face_records({execution, record_count});
}

}  // namespace

thrust::device_vector<Triangle>& coarsened_dense_mesh_cuda(
    const CudaRiFrame& frame,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    DenseMeshOptions options
)
{
    if (frame.macro_quads == nullptr) {
        throw std::runtime_error("coarsened filled-pixel mode has no macro-quad workspace");
    }
    CudaWorkspace& workspace = execution.workspace;
    const int quad_count = (frame.height - 1) * frame.width;
    resize_with_headroom(workspace.quad_face_flags, static_cast<std::size_t>(quad_count));
    // Compute the regular 1x1 decisions first. Macro cells may replace those
    // decisions, but fine fallback cells retain every existing dense-mesh
    // filter, including max_triangle_dist and isolated-component filtering.
    mark_dense_candidate_faces({frame, config, execution, options});
    filter_isolated_component_slots_cuda({frame, config, execution});
    build_macro_quad_plan(frame, config, execution, options);
    return emit_coarsened_faces(frame, config, execution, options);
}

}  // namespace ri_meshing::cuda_detail
