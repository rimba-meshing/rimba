#include "dense_meshing_internal.cuh"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <type_traits>

namespace ri_meshing::cuda_detail {
namespace {

struct FaceRecordLess {
    __host__ __device__ bool operator()(const FaceRecord& a, const FaceRecord& b) const
    {
        if (a.key0 != b.key0) {
            return a.key0 < b.key0;
        }
        if (a.key1 != b.key1) {
            return a.key1 < b.key1;
        }
        return a.key2 < b.key2;
    }
};

struct UniqueFaceInput {
    const FaceRecord* records;
    int* flags;
};

struct FaceCompactionInput {
    const FaceRecord* records;
    const int* offsets;
    const int* flags;
    Triangle* faces;
};

static_assert(std::is_trivially_copyable_v<UniqueFaceInput>);
static_assert(std::is_trivially_copyable_v<FaceCompactionInput>);

__global__ void mark_unique_faces_kernel(UniqueFaceInput input, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const bool same_prev = idx > 0
        && input.records[idx - 1].key0 == input.records[idx].key0
        && input.records[idx - 1].key1 == input.records[idx].key1
        && input.records[idx - 1].key2 == input.records[idx].key2;
    const bool same_next = idx + 1 < count
        && input.records[idx + 1].key0 == input.records[idx].key0
        && input.records[idx + 1].key1 == input.records[idx].key1
        && input.records[idx + 1].key2 == input.records[idx].key2;
    input.flags[idx] = !(same_prev || same_next);
}

__global__ void compact_faces_kernel(FaceCompactionInput input, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && input.flags[idx] != 0) {
        input.faces[input.offsets[idx]] = input.records[idx].face;
    }
}

}  // namespace

thrust::device_vector<Triangle>& compact_dense_face_records(DenseCompactionRequest request)
{
    CudaWorkspace& workspace = request.execution.workspace;
    const int valid_record_count = request.valid_record_count;
    constexpr int block = 256;
    const auto grid_for = [](int count) { return (count + block - 1) / block; };

    timed_cuda_stage_void(request.execution.timing, "mesh.sort_face_keys", [&] {
        thrust::sort(
            thrust::cuda::par.on(request.execution.stream),
            workspace.face_records.begin(),
            workspace.face_records.begin() + valid_record_count,
            FaceRecordLess{}
        );
    });

    resize_with_headroom(workspace.flags, static_cast<std::size_t>(valid_record_count));
    resize_with_headroom(workspace.offsets, static_cast<std::size_t>(valid_record_count));
    int face_count = 0;
    timed_cuda_stage_void(request.execution.timing, "mesh.unique_prefix_sum", [&] {
        mark_unique_faces_kernel<<<grid_for(valid_record_count), block, 0, request.execution.stream>>>(
            UniqueFaceInput{
                thrust::raw_pointer_cast(workspace.face_records.data()),
                thrust::raw_pointer_cast(workspace.flags.data()),
            },
            valid_record_count
        );
        CUDA_CHECK(cudaGetLastError());
        face_count = scanned_count(workspace.flags, workspace.offsets, request.execution.stream);
    });

    resize_with_headroom(workspace.faces, static_cast<std::size_t>(face_count));
    timed_cuda_stage_void(request.execution.timing, "mesh.compact_faces", [&] {
        compact_faces_kernel<<<grid_for(valid_record_count), block, 0, request.execution.stream>>>(
            FaceCompactionInput{
                thrust::raw_pointer_cast(workspace.face_records.data()),
                thrust::raw_pointer_cast(workspace.offsets.data()),
                thrust::raw_pointer_cast(workspace.flags.data()),
                thrust::raw_pointer_cast(workspace.faces.data()),
            },
            valid_record_count
        );
    });
    return workspace.faces;
}

}  // namespace ri_meshing::cuda_detail
