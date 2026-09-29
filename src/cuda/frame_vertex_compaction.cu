#include "cuda_workflow.hpp"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <thrust/execution_policy.h>

#include <stdexcept>
#include <type_traits>

namespace ri_meshing::cuda_detail {
namespace {

struct MarkUsedVerticesView {
    const Triangle* faces{};
    int* used{};
    int old_vertex_count{};
    int new_vertex_count{};
};

struct CompactVerticesView {
    const Vec3f* input{};
    Vec3f* output{};
    const int* used{};
    const int* offsets{};
    int old_vertex_count{};
};

struct RemapFacesView {
    Triangle* faces{};
    const int* used{};
    const int* offsets{};
    int old_vertex_count{};
    int new_vertex_count{};
};

struct RemapVimView {
    int* vim{};
    const int* used{};
    const int* offsets{};
    int old_vertex_count{};
    int new_vertex_count{};
};

struct RebaseVerticesView {
    const Vec3f* input{};
    Vec3f* output{};
    int old_vertex_count{};
    int new_old_vertex_count{};
};

static_assert(std::is_trivially_copyable_v<MarkUsedVerticesView>);
static_assert(std::is_trivially_copyable_v<CompactVerticesView>);
static_assert(std::is_trivially_copyable_v<RemapFacesView>);
static_assert(std::is_trivially_copyable_v<RemapVimView>);
static_assert(std::is_trivially_copyable_v<RebaseVerticesView>);

constexpr int kBlock = 256;

int grid_for(int count)
{
    return (count + kBlock - 1) / kBlock;
}

__global__ void mark_used_vertices_kernel(MarkUsedVerticesView view, int face_count)
{
    const int face_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_index >= face_count) {
        return;
    }
    const Triangle face = view.faces[face_index];
    const int ids[3] = {face.v0, face.v1, face.v2};
    for (const int id : ids) {
        const int local_id = id - view.old_vertex_count;
        if (local_id >= 0 && local_id < view.new_vertex_count) {
            atomicExch(&view.used[local_id], 1);
        }
    }
}

__global__ void compact_vertices_kernel(CompactVerticesView view, int new_vertex_count)
{
    const int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_id >= new_vertex_count || view.used[local_id] == 0) {
        return;
    }
    view.output[view.old_vertex_count + view.offsets[local_id]]
        = view.input[view.old_vertex_count + local_id];
}

__global__ void remap_faces_kernel(RemapFacesView view, int face_count)
{
    const int face_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_index >= face_count) {
        return;
    }
    Triangle& face = view.faces[face_index];
    int* ids[3] = {&face.v0, &face.v1, &face.v2};
    for (int* id : ids) {
        const int local_id = *id - view.old_vertex_count;
        if (local_id < 0) {
            continue;
        }
        if (local_id >= view.new_vertex_count || view.used[local_id] == 0) {
            *id = -1;
            continue;
        }
        *id = view.old_vertex_count + view.offsets[local_id];
    }
}

__global__ void remap_vim_kernel(RemapVimView view, int pixel_count)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const int local_id = view.vim[pixel] - view.old_vertex_count;
    if (local_id < 0) {
        return;
    }
    if (local_id >= view.new_vertex_count || view.used[local_id] == 0) {
        view.vim[pixel] = -1;
        return;
    }
    view.vim[pixel] = view.old_vertex_count + view.offsets[local_id];
}

__global__ void rebase_vertices_kernel(
    RebaseVerticesView view,
    int new_vertex_count
)
{
    const int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_id >= new_vertex_count) {
        return;
    }
    view.output[view.new_old_vertex_count + local_id]
        = view.input[view.old_vertex_count + local_id];
}

__device__ void rebase_id(
    int* id,
    int old_vertex_count,
    int new_old_vertex_count,
    int new_vertex_count
)
{
    if (*id >= old_vertex_count
        && *id < old_vertex_count + new_vertex_count) {
        *id += new_old_vertex_count - old_vertex_count;
    }
}

__global__ void rebase_faces_kernel(
    Triangle* faces,
    int face_count,
    int old_vertex_count,
    int new_old_vertex_count,
    int new_vertex_count
)
{
    const int face_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_index >= face_count) {
        return;
    }
    Triangle& face = faces[face_index];
    rebase_id(&face.v0, old_vertex_count, new_old_vertex_count, new_vertex_count);
    rebase_id(&face.v1, old_vertex_count, new_old_vertex_count, new_vertex_count);
    rebase_id(&face.v2, old_vertex_count, new_old_vertex_count, new_vertex_count);
}

__global__ void rebase_vim_kernel(
    int* vim,
    int pixel_count,
    int old_vertex_count,
    int new_old_vertex_count,
    int new_vertex_count
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    rebase_id(&vim[pixel], old_vertex_count, new_old_vertex_count, new_vertex_count);
}

int mark_and_scan_used_vertices(
    CudaRiFrame& frame,
    thrust::device_vector<Triangle>& faces,
    CudaExecutionContext execution
)
{
    CudaWorkspace& workspace = execution.workspace;
    resize_with_headroom(workspace.flags, static_cast<std::size_t>(frame.new_vertex_count));
    resize_with_headroom(workspace.offsets, static_cast<std::size_t>(frame.new_vertex_count));
    thrust::fill(
        thrust::cuda::par.on(execution.stream),
        workspace.flags.begin(),
        workspace.flags.end(),
        0
    );
    if (!faces.empty()) {
        mark_used_vertices_kernel<<<grid_for(static_cast<int>(faces.size())), kBlock, 0, execution.stream>>>(
            MarkUsedVerticesView{
                thrust::raw_pointer_cast(faces.data()),
                thrust::raw_pointer_cast(workspace.flags.data()),
                frame.old_vertex_count,
                frame.new_vertex_count,
            },
            static_cast<int>(faces.size())
        );
        CUDA_CHECK(cudaGetLastError());
    }
    return scanned_count(workspace.flags, workspace.offsets, execution.stream);
}

void remap_compacted_frame(
    CudaRiFrame& frame,
    thrust::device_vector<Triangle>& faces,
    int used_count,
    CudaExecutionContext execution
)
{
    CudaWorkspace& workspace = execution.workspace;
    resize_with_headroom(
        workspace.compacted_current_vertices,
        static_cast<std::size_t>(frame.old_vertex_count + used_count)
    );
    if (used_count > 0) {
        compact_vertices_kernel<<<grid_for(frame.new_vertex_count), kBlock, 0, execution.stream>>>(
            CompactVerticesView{
                thrust::raw_pointer_cast(frame.vertices->data()),
                thrust::raw_pointer_cast(workspace.compacted_current_vertices.data()),
                thrust::raw_pointer_cast(workspace.flags.data()),
                thrust::raw_pointer_cast(workspace.offsets.data()),
                frame.old_vertex_count,
            },
            frame.new_vertex_count
        );
        CUDA_CHECK(cudaGetLastError());
    }
    if (!faces.empty()) {
        remap_faces_kernel<<<grid_for(static_cast<int>(faces.size())), kBlock, 0, execution.stream>>>(
            RemapFacesView{
                thrust::raw_pointer_cast(faces.data()),
                thrust::raw_pointer_cast(workspace.flags.data()),
                thrust::raw_pointer_cast(workspace.offsets.data()),
                frame.old_vertex_count,
                frame.new_vertex_count,
            },
            static_cast<int>(faces.size())
        );
        CUDA_CHECK(cudaGetLastError());
    }
    if (frame.filled_vim != nullptr && !frame.filled_vim->empty()) {
        remap_vim_kernel<<<grid_for(static_cast<int>(frame.filled_vim->size())), kBlock, 0, execution.stream>>>(
            RemapVimView{
                thrust::raw_pointer_cast(frame.filled_vim->data()),
                thrust::raw_pointer_cast(workspace.flags.data()),
                thrust::raw_pointer_cast(workspace.offsets.data()),
                frame.old_vertex_count,
                frame.new_vertex_count,
            },
            static_cast<int>(frame.filled_vim->size())
        );
        CUDA_CHECK(cudaGetLastError());
    }
    workspace.current_vertices_local.swap(workspace.compacted_current_vertices);
    frame.new_vertex_count = used_count;
}

}  // namespace

void compact_current_frame_vertices_cuda(
    CudaRiFrame& frame,
    thrust::device_vector<Triangle>& faces,
    CudaExecutionContext execution
)
{
    if (frame.new_vertex_count <= 0) {
        return;
    }
    const int used_count = mark_and_scan_used_vertices(frame, faces, execution);
    if (used_count != frame.new_vertex_count) {
        remap_compacted_frame(frame, faces, used_count, execution);
    }
}

void rebase_prefetched_frame_cuda(
    CudaRiFrame& frame,
    thrust::device_vector<Triangle>& faces,
    int new_old_vertex_count,
    CudaExecutionContext execution
)
{
    if (frame.new_vertex_count <= 0
        || new_old_vertex_count == frame.old_vertex_count) {
        frame.old_vertex_count = new_old_vertex_count;
        return;
    }
    if (new_old_vertex_count < frame.old_vertex_count) {
        throw std::runtime_error("prefetched frame vertex base moved backwards");
    }
    const int old_vertex_count = frame.old_vertex_count;
    const int new_vertex_count = frame.new_vertex_count;
    CudaWorkspace& workspace = execution.workspace;
    timed_cuda_stage_void(execution.timing, "overlap.rebase_prefetched_frame", [&] {
        resize_with_headroom(
            workspace.compacted_current_vertices,
            static_cast<std::size_t>(new_old_vertex_count + new_vertex_count)
        );
        rebase_vertices_kernel<<<grid_for(new_vertex_count), kBlock, 0, execution.stream>>>(
            RebaseVerticesView{
                thrust::raw_pointer_cast(frame.vertices->data()),
                thrust::raw_pointer_cast(workspace.compacted_current_vertices.data()),
                old_vertex_count,
                new_old_vertex_count,
            },
            new_vertex_count
        );
        CUDA_CHECK(cudaGetLastError());
        if (!faces.empty()) {
            rebase_faces_kernel<<<grid_for(static_cast<int>(faces.size())), kBlock, 0, execution.stream>>>(
                thrust::raw_pointer_cast(faces.data()),
                static_cast<int>(faces.size()),
                old_vertex_count,
                new_old_vertex_count,
                new_vertex_count
            );
            CUDA_CHECK(cudaGetLastError());
        }
        if (frame.filled_vim != nullptr && !frame.filled_vim->empty()) {
            rebase_vim_kernel<<<grid_for(static_cast<int>(frame.filled_vim->size())), kBlock, 0, execution.stream>>>(
                thrust::raw_pointer_cast(frame.filled_vim->data()),
                static_cast<int>(frame.filled_vim->size()),
                old_vertex_count,
                new_old_vertex_count,
                new_vertex_count
            );
            CUDA_CHECK(cudaGetLastError());
        }
    });
    workspace.current_vertices_local.swap(workspace.compacted_current_vertices);
    frame.old_vertex_count = new_old_vertex_count;
}

}  // namespace ri_meshing::cuda_detail
