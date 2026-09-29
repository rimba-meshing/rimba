#include "cuda_workflow.hpp"

#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <thrust/host_vector.h>

int main()
{
    using ri_meshing::Vec3f;
    using ri_meshing::cuda_detail::AccumulatedFrameChunk;
    using ri_meshing::cuda_detail::AccumulatedFaceRange;
    using ri_meshing::cuda_detail::AccumulatedVertexRange;
    using ri_meshing::cuda_detail::accumulated_chunk_intersects_radius;

    const AccumulatedFrameChunk chunk{
        0, 10, 0, 20, Vec3f{74.0F, 0.0F, 0.0F}, 4.0F
    };
    if (!accumulated_chunk_intersects_radius(chunk, Vec3f{}, 70.0F)) {
        return 1;
    }

    const AccumulatedFrameChunk outside{
        0, 10, 0, 20, Vec3f{75.0F, 0.0F, 0.0F}, 4.0F
    };
    if (accumulated_chunk_intersects_radius(outside, Vec3f{}, 70.0F)) {
        return 1;
    }
    if (!accumulated_chunk_intersects_radius(outside, Vec3f{9.0F, 0.0F, 0.0F}, 70.0F)) {
        return 1;
    }

    ri_meshing::SparseMeshingConfig config{};
    config.ri_height = 8;
    config.ri_width = 8;
    config.vertical_fov.min_rad = -1.5707963F;
    config.vertical_fov.max_rad = 1.5707963F;
    ri_meshing::cuda_detail::CudaWorkspace workspace;
    workspace.accumulated_vertices_global = {
        Vec3f{10.0F, -0.1F, -0.1F},
        Vec3f{10.0F, 0.1F, -0.1F},
        Vec3f{10.0F, 0.0F, 0.1F},
        Vec3f{100.0F, -0.1F, -0.1F},
        Vec3f{100.0F, 0.1F, -0.1F},
        Vec3f{100.0F, 0.0F, 0.1F},
    };
    workspace.accumulated_faces_global = {
        ri_meshing::Triangle{0, 1, 2},
        ri_meshing::Triangle{3, 4, 5},
    };
    const std::vector<AccumulatedFrameChunk> chunks{
        AccumulatedFrameChunk{0, 3, 0, 1, Vec3f{10.0F, 0.0F, 0.0F}, 1.0F},
        AccumulatedFrameChunk{3, 3, 1, 1, Vec3f{100.0F, 0.0F, 0.0F}, 1.0F},
    };
    ri_meshing::cuda_detail::prepare_accumulated_depth_cuda(
        config,
        ri_meshing::cuda_detail::CudaExecutionContext{workspace, nullptr, 0},
        ri_meshing::cuda_detail::AccumulatedDepthOptions{
            ri_meshing::cuda_detail::CudaAccumulationCounts{6, 2},
            ri_meshing::Mat4f{1.0F, 0.0F, 0.0F, 0.0F,
                              0.0F, 1.0F, 0.0F, 0.0F,
                              0.0F, 0.0F, 1.0F, 0.0F,
                              0.0F, 0.0F, 0.0F, 1.0F},
            &chunks,
        }
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    if (workspace.active_face_ranges.size() != 1) {
        return 1;
    }
    const thrust::host_vector<AccumulatedFaceRange> active_ranges = workspace.active_face_ranges;
    if (active_ranges.front().first_face != 0 || active_ranges.front().face_count != 1) {
        return 1;
    }
    const thrust::host_vector<AccumulatedVertexRange> active_vertex_ranges =
        workspace.active_vertex_ranges;
    if (active_vertex_ranges.size() != 1
        || active_vertex_ranges.front().first_vertex != 0
        || active_vertex_ranges.front().vertex_count != 3) {
        return 1;
    }

    const std::vector<AccumulatedFrameChunk> all_far_chunks{
        AccumulatedFrameChunk{0, 3, 0, 1, Vec3f{100.0F, 0.0F, 0.0F}, 1.0F},
        AccumulatedFrameChunk{3, 3, 1, 1, Vec3f{120.0F, 0.0F, 0.0F}, 1.0F},
    };
    ri_meshing::cuda_detail::prepare_accumulated_depth_cuda(
        config,
        ri_meshing::cuda_detail::CudaExecutionContext{workspace, nullptr, 0},
        ri_meshing::cuda_detail::AccumulatedDepthOptions{
            ri_meshing::cuda_detail::CudaAccumulationCounts{6, 2},
            ri_meshing::Mat4f{1.0F, 0.0F, 0.0F, 0.0F,
                              0.0F, 1.0F, 0.0F, 0.0F,
                              0.0F, 0.0F, 1.0F, 0.0F,
                              0.0F, 0.0F, 0.0F, 1.0F},
            &all_far_chunks,
        }
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!workspace.active_face_ranges.empty()) {
        return 1;
    }
    if (!workspace.active_vertex_ranges.empty()) {
        return 1;
    }

    workspace.accumulated_vertices_global.resize(7);
    workspace.accumulated_vertices_global[6] = Vec3f{10.0F, 0.0F, 0.0F};
    ri_meshing::cuda_detail::append_accumulated_depth_cuda(
        config,
        ri_meshing::cuda_detail::CudaExecutionContext{workspace, nullptr, 0},
        ri_meshing::cuda_detail::AccumulatedDepthAppendOptions{
            ri_meshing::cuda_detail::CudaAccumulationCounts{7, 2},
            ri_meshing::Mat4f{1.0F, 0.0F, 0.0F, 0.0F,
                              0.0F, 1.0F, 0.0F, 0.0F,
                              0.0F, 0.0F, 1.0F, 0.0F,
                              0.0F, 0.0F, 0.0F, 1.0F},
            6,
            1,
            1,
            1,
        }
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    const thrust::host_vector<AccumulatedFaceRange> appended_face_ranges =
        workspace.active_face_ranges;
    if (appended_face_ranges.size() != 1
        || appended_face_ranges.front().first_face != 1
        || appended_face_ranges.front().face_count != 1) {
        return 1;
    }
    const thrust::host_vector<AccumulatedVertexRange> appended_vertex_ranges =
        workspace.active_vertex_ranges;
    if (appended_vertex_ranges.size() != 1
        || appended_vertex_ranges.front().first_vertex != 6
        || appended_vertex_ranges.front().vertex_count != 1) {
        return 1;
    }
    return 0;
}
