#include "cuda_workflow.hpp"

#include "accumulation_transform.cuh"
#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <thrust/device_vector.h>

#include <type_traits>

namespace ri_meshing::cuda_detail {
namespace {

using accumulation_internal::DeviceMat4;
using accumulation_internal::to_device_mat;
using accumulation_internal::transform_point;

struct VertexAppendView {
    const Vec3f* local;
    Vec3f* global;
    int first;
    int count;
};

static_assert(std::is_trivially_copyable_v<VertexAppendView>);

__global__ void transform_new_vertices_to_global_kernel(
    VertexAppendView vertices,
    DeviceMat4 pose
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vertices.count) {
        vertices.global[vertices.first + idx] = transform_point(vertices.local[vertices.first + idx], pose);
    }
}

struct AccumulationAppendRequest {
    CudaWorkspace* workspace;
    CudaAccumulationCounts* counts;
    const CudaRiFrame* frame;
    const thrust::device_vector<Triangle>* faces;
    const Mat4f* pose;
    PipelineTiming* timing;
};

class AccumulationAppendPass {
public:
    explicit AccumulationAppendPass(AccumulationAppendRequest request)
        : request_(request),
          old_vertex_count_(request.counts->vertex_count),
          old_face_count_(request.counts->face_count),
          new_vertex_count_(request.frame->new_vertex_count),
          new_face_count_(static_cast<int>(request.faces->size())),
          local_to_global_(to_device_mat(*request.pose))
    {
    }

    void run()
    {
        append_vertices();
        request_.counts->vertex_count += new_vertex_count_;
        append_faces();
        request_.counts->face_count += new_face_count_;
    }

private:
    static constexpr int kBlockSize = 256;

    static int grid_for(int count)
    {
        return (count + kBlockSize - 1) / kBlockSize;
    }

    void append_vertices()
    {
        timed_cuda_stage_void(request_.timing, "accum.append_vertices_global", [&] {
            resize_with_headroom(
                request_.workspace->accumulated_vertices_global,
                static_cast<std::size_t>(old_vertex_count_ + new_vertex_count_)
            );
            if (new_vertex_count_ > 0) {
                launch_vertex_append();
            }
        });
    }

    void launch_vertex_append()
    {
        transform_new_vertices_to_global_kernel<<<grid_for(new_vertex_count_), kBlockSize>>>(
            {
                thrust::raw_pointer_cast(request_.frame->vertices->data()),
                thrust::raw_pointer_cast(request_.workspace->accumulated_vertices_global.data()),
                old_vertex_count_,
                new_vertex_count_
            },
            local_to_global_
        );
    }

    void append_faces()
    {
        timed_cuda_stage_void(request_.timing, "accum.append_faces_global", [&] {
            resize_with_headroom(
                request_.workspace->accumulated_faces_global,
                static_cast<std::size_t>(old_face_count_ + new_face_count_)
            );
            if (new_face_count_ > 0) {
                copy_device_to_device_async(
                    thrust::raw_pointer_cast(
                        request_.workspace->accumulated_faces_global.data()
                    ) + old_face_count_,
                    thrust::raw_pointer_cast(request_.faces->data()),
                    static_cast<std::size_t>(new_face_count_)
                );
            }
        });
    }

    AccumulationAppendRequest request_;
    int old_vertex_count_;
    int old_face_count_;
    int new_vertex_count_;
    int new_face_count_;
    DeviceMat4 local_to_global_;
};

}  // namespace

void append_current_frame_to_accumulation_cuda(
    CudaAccumulationCounts& counts,
    CudaExecutionContext execution,
    AccumulationAppendInput input
)
{
    AccumulationAppendPass({
        &execution.workspace,
        &counts,
        &input.frame,
        &input.faces,
        &input.pose,
        execution.timing,
    }).run();
}

}  // namespace ri_meshing::cuda_detail
