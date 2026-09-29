#include "dense_meshing_internal.cuh"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <stdexcept>
#include <type_traits>

namespace ri_meshing::cuda_detail {
namespace {

constexpr int kMaxGpuIsolatedComponentFaces = 64;
constexpr unsigned char kQuadFaceSlotA = 1U;
constexpr unsigned char kQuadFaceSlotB = 2U;
constexpr unsigned char kQuadFaceAlternate = 4U;

struct RiDimensions {
    int height;
    int width;
};

struct ComponentSlotView {
    const int* vim;
    const unsigned char* quad_face_flags;
    int* keep_flags;
};

struct ComponentFilterParams {
    RiDimensions dimensions;
    int max_faces;
};

static_assert(std::is_trivially_copyable_v<RiDimensions>);
static_assert(std::is_trivially_copyable_v<ComponentSlotView>);
static_assert(std::is_trivially_copyable_v<ComponentFilterParams>);

__device__ bool quad_slot_valid(const unsigned char* quad_face_flags, int quad, int slot)
{
    const unsigned char flags = quad_face_flags[quad];
    const unsigned char mask = slot == 0 ? kQuadFaceSlotA : kQuadFaceSlotB;
    return (flags & mask) != 0;
}

__device__ Triangle quad_slot_face(
    const int* vim,
    const unsigned char* quad_face_flags,
    int quad,
    int slot,
    int width
)
{
    const int row = quad / width;
    const int col = quad - row * width;
    const int next_col = (col + 1) % width;
    const int tl = vim[row * width + col];
    const int tr = vim[row * width + next_col];
    const int bl = vim[(row + 1) * width + col];
    const int br = vim[(row + 1) * width + next_col];
    const bool alternate = (quad_face_flags[quad] & kQuadFaceAlternate) != 0;
    if (alternate) {
        return slot == 0 ? Triangle{tl, bl, br} : Triangle{tl, br, tr};
    }
    return slot == 0 ? Triangle{tl, bl, tr} : Triangle{tr, bl, br};
}

__device__ bool contains_slot(const int* slots, int count, int slot)
{
    for (int idx = 0; idx < count; ++idx) {
        if (slots[idx] == slot) {
            return true;
        }
    }
    return false;
}

__device__ bool faces_share_edge(Triangle a, Triangle b)
{
    int shared = 0;
    shared += static_cast<int>(a.v0 == b.v0 || a.v0 == b.v1 || a.v0 == b.v2);
    shared += static_cast<int>(a.v1 == b.v0 || a.v1 == b.v1 || a.v1 == b.v2);
    shared += static_cast<int>(a.v2 == b.v0 || a.v2 == b.v1 || a.v2 == b.v2);
    return shared >= 2;
}

__global__ void mark_large_quad_slot_components_kernel(
    ComponentSlotView slots,
    ComponentFilterParams params
)
{
    const int height = params.dimensions.height;
    const int width = params.dimensions.width;
    const int start_slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int quad_count = (height - 1) * width;
    const int slot_count = quad_count * 2;
    if (start_slot >= slot_count) {
        return;
    }
    const int start_quad = start_slot / 2;
    const int start_local_slot = start_slot & 1;
    if (!quad_slot_valid(slots.quad_face_flags, start_quad, start_local_slot)) {
        slots.keep_flags[start_slot] = 0;
        return;
    }

    int visited[kMaxGpuIsolatedComponentFaces + 1];
    int queue[kMaxGpuIsolatedComponentFaces + 1];
    int visited_count = 1;
    int queue_begin = 0;
    int queue_end = 1;
    visited[0] = start_slot;
    queue[0] = start_slot;

    while (queue_begin < queue_end) {
        const int slot_idx = queue[queue_begin++];
        const int quad = slot_idx / 2;
        const int local_slot = slot_idx & 1;
        const int row = quad / width;
        const int col = quad - row * width;
        const Triangle face = quad_slot_face(slots.vim, slots.quad_face_flags, quad, local_slot, width);

        for (int dr = -1; dr <= 1; ++dr) {
            const int nr = row + dr;
            if (nr < 0 || nr >= height - 1) {
                continue;
            }
            for (int dc = -1; dc <= 1; ++dc) {
                int nc = col + dc;
                if (nc < 0) {
                    nc += width;
                } else if (nc >= width) {
                    nc -= width;
                }
                const int neighbor_quad = nr * width + nc;
                for (int neighbor_local_slot = 0; neighbor_local_slot < 2; ++neighbor_local_slot) {
                    if (!quad_slot_valid(slots.quad_face_flags, neighbor_quad, neighbor_local_slot)) {
                        continue;
                    }
                    const int neighbor_slot = neighbor_quad * 2 + neighbor_local_slot;
                    if (neighbor_slot == slot_idx || contains_slot(visited, visited_count, neighbor_slot)) {
                        continue;
                    }
                    const Triangle neighbor_face = quad_slot_face(
                        slots.vim,
                        slots.quad_face_flags,
                        neighbor_quad,
                        neighbor_local_slot,
                        width
                    );
                    if (!faces_share_edge(face, neighbor_face)) {
                        continue;
                    }
                    visited[visited_count] = neighbor_slot;
                    queue[queue_end] = neighbor_slot;
                    ++visited_count;
                    ++queue_end;
                    if (visited_count > params.max_faces) {
                        slots.keep_flags[start_slot] = 1;
                        return;
                    }
                }
            }
        }
    }

    slots.keep_flags[start_slot] = 0;
}

}  // namespace

void filter_isolated_component_slots_cuda(DenseIsolationRequest request)
{
    if (request.config.dense_isolated_component_max_faces <= 0) {
        return;
    }
    if (request.config.dense_isolated_component_max_faces > kMaxGpuIsolatedComponentFaces) {
        throw std::runtime_error("dense isolated component GPU filter supports limits up to 64 faces");
    }

    constexpr int block = 256;
    const auto grid_for = [](int count) { return (count + block - 1) / block; };
    const int quad_count = (request.frame.height - 1) * request.frame.width;
    const int slot_count = quad_count * 2;
    timed_cuda_stage_void(request.execution.timing, "mesh.filter_isolated_components", [&] {
        resize_with_headroom(request.execution.workspace.flags, static_cast<std::size_t>(slot_count));
        mark_large_quad_slot_components_kernel<<<grid_for(slot_count), block, 0, request.execution.stream>>>(
            ComponentSlotView{
                thrust::raw_pointer_cast(request.frame.filled_vim->data()),
                thrust::raw_pointer_cast(request.execution.workspace.quad_face_flags.data()),
                thrust::raw_pointer_cast(request.execution.workspace.flags.data()),
            },
            ComponentFilterParams{
                {request.frame.height, request.frame.width},
                request.config.dense_isolated_component_max_faces,
            }
        );
        CUDA_CHECK(cudaGetLastError());
    });
}

}  // namespace ri_meshing::cuda_detail
