#include "stitch_geometry_cuda.cuh"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"
#include "cuda_workflow.hpp"

#include <cmath>
#include <stdexcept>

namespace ri_meshing::cuda_pipeline {

Mat4f inverse_rigid_cpu(Mat4f pose)
{
    Mat4f out{static_cast<MeshScalar>(1), static_cast<MeshScalar>(0), static_cast<MeshScalar>(0), static_cast<MeshScalar>(0),
              static_cast<MeshScalar>(0), static_cast<MeshScalar>(1), static_cast<MeshScalar>(0), static_cast<MeshScalar>(0),
              static_cast<MeshScalar>(0), static_cast<MeshScalar>(0), static_cast<MeshScalar>(1), static_cast<MeshScalar>(0),
              static_cast<MeshScalar>(0), static_cast<MeshScalar>(0), static_cast<MeshScalar>(0), static_cast<MeshScalar>(1)};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out[row * 4 + col] = pose[col * 4 + row];
        }
    }
    for (int row = 0; row < 3; ++row) {
        out[row * 4 + 3] = -(
            out[row * 4] * pose[3] + out[row * 4 + 1] * pose[7] + out[row * 4 + 2] * pose[11]
        );
    }
    return out;
}

Vec3f transform_point_cpu(Vec3f point, const Mat4f& transform)
{
    const MeshScalar x = transform[0] * point.x + transform[1] * point.y + transform[2] * point.z + transform[3];
    const MeshScalar y = transform[4] * point.x + transform[5] * point.y + transform[6] * point.z + transform[7];
    const MeshScalar z = transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11];
    const MeshScalar w = transform[12] * point.x + transform[13] * point.y + transform[14] * point.z + transform[15];
    const MeshScalar inv_w = std::abs(w) > static_cast<MeshScalar>(1e-12)
        ? static_cast<MeshScalar>(1) / w : static_cast<MeshScalar>(1);
    return Vec3f{static_cast<float>(x * inv_w), static_cast<float>(y * inv_w), static_cast<float>(z * inv_w)};
}

void sync_new_host_vertices(
    cuda_detail::CudaWorkspace& workspace,
    std::vector<Vec3f>& host_vertices_global,
    int old_vertex_count,
    int new_vertex_count,
    PipelineTiming* timing
)
{
    cuda_detail::timed_cuda_stage_void(timing, "stitch.sync_host_vertices", [&] {
        if (new_vertex_count <= 0) {
            return;
        }
        if (old_vertex_count < 0) {
            throw std::runtime_error("negative old vertex count while syncing stitching host vertices");
        }
        const std::size_t first_new_vertex = static_cast<std::size_t>(old_vertex_count);
        const Stopwatch resize_stopwatch;
        host_vertices_global.resize(first_new_vertex + static_cast<std::size_t>(new_vertex_count));
        if (timing != nullptr) {
            timing->add_stage(
                "stitch.sync_host_vertices.resize_host_buffer",
                resize_stopwatch.elapsed_milliseconds()
            );
        }
        const Stopwatch download_stopwatch;
        cuda_detail::copy_device_to_host_sync(
            host_vertices_global.data() + old_vertex_count,
            thrust::raw_pointer_cast(workspace.accumulated_vertices_global.data()) + old_vertex_count,
            static_cast<std::size_t>(new_vertex_count)
        );
        if (timing != nullptr) {
            timing->add_stage(
                "stitch.sync_host_vertices.download_vertices",
                download_stopwatch.elapsed_milliseconds()
            );
        }
    });
}

namespace {

struct FaceNormalAccumulationView {
    const Triangle* faces{};
    const Vec3f* vertices{};
    Vec3f* normal_sums{};
    int face_count{};
    int first_new_vertex{};
    int new_vertex_count{};
};

constexpr int kNormalAccumulationBlockSize = 256;

int normal_accumulation_grid_for(int count)
{
    return (count + kNormalAccumulationBlockSize - 1) / kNormalAccumulationBlockSize;
}

__device__ void atomic_add_normal(Vec3f* destination, Vec3f value)
{
    atomicAdd(&destination->x, value.x);
    atomicAdd(&destination->y, value.y);
    atomicAdd(&destination->z, value.z);
}

__global__ void accumulate_face_normal_sums_kernel(FaceNormalAccumulationView view)
{
    const int face_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_index >= view.face_count) {
        return;
    }
    const Triangle face = view.faces[face_index];
    const int first = view.first_new_vertex;
    const int last = first + view.new_vertex_count;
    if (face.v0 < first || face.v0 >= last
        || face.v1 < first || face.v1 >= last
        || face.v2 < first || face.v2 >= last) {
        return;
    }
    const Vec3f a = view.vertices[face.v0];
    const Vec3f b = view.vertices[face.v1];
    const Vec3f c = view.vertices[face.v2];
    const Vec3f ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3f ac{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3f normal{
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x,
    };
    atomic_add_normal(&view.normal_sums[face.v0], normal);
    atomic_add_normal(&view.normal_sums[face.v1], normal);
    atomic_add_normal(&view.normal_sums[face.v2], normal);
}

void accumulate_face_normal_sums_impl(
    const std::vector<Vec3f>& vertices_global,
    const std::vector<Triangle>& faces,
    std::vector<Vec3f>& normal_sums_global,
    PipelineTiming* timing
)
{
    const Stopwatch stopwatch;
    normal_sums_global.resize(vertices_global.size(), Vec3f{});
    for (const Triangle face : faces) {
        if (face.v0 < 0 || face.v1 < 0 || face.v2 < 0
            || static_cast<std::size_t>(face.v0) >= vertices_global.size()
            || static_cast<std::size_t>(face.v1) >= vertices_global.size()
            || static_cast<std::size_t>(face.v2) >= vertices_global.size()) {
            continue;
        }
        const Vec3f a = vertices_global[static_cast<std::size_t>(face.v0)];
        const Vec3f b = vertices_global[static_cast<std::size_t>(face.v1)];
        const Vec3f c = vertices_global[static_cast<std::size_t>(face.v2)];
        const Vec3f ab{b.x - a.x, b.y - a.y, b.z - a.z};
        const Vec3f ac{c.x - a.x, c.y - a.y, c.z - a.z};
        const Vec3f normal{
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x,
        };
        for (const int vertex : {face.v0, face.v1, face.v2}) {
            Vec3f& sum = normal_sums_global[static_cast<std::size_t>(vertex)];
            sum.x += normal.x;
            sum.y += normal.y;
            sum.z += normal.z;
        }
    }
    if (timing != nullptr) {
        timing->add_stage("stitch.incremental_normals.accumulate", stopwatch.elapsed_milliseconds());
    }
}

}  // namespace

void launch_device_face_normal_sums(DeviceNormalAccumulationRequest request)
{
    if (request.first_new_vertex < 0 || request.new_vertex_count < 0) {
        throw std::runtime_error("invalid current vertex range while accumulating normals");
    }
    const std::size_t required_size = static_cast<std::size_t>(request.first_new_vertex)
        + static_cast<std::size_t>(request.new_vertex_count);
    request.normal_sums_global.resize(required_size, Vec3f{});
    if (request.new_vertex_count == 0) {
        return;
    }

    cuda_detail::resize_with_headroom(
        request.workspace.current_vertex_normal_sums,
        required_size
    );
    cuda_detail::timed_cuda_stage_void(request.timing, "stitch.incremental_normals.gpu_accumulate", [&] {
        CUDA_CHECK(cudaMemsetAsync(
            thrust::raw_pointer_cast(request.workspace.current_vertex_normal_sums.data())
                + request.first_new_vertex,
            0,
            static_cast<std::size_t>(request.new_vertex_count) * sizeof(Vec3f),
            request.stream
        ));
        if (!request.faces.empty()) {
            accumulate_face_normal_sums_kernel<<<
                normal_accumulation_grid_for(static_cast<int>(request.faces.size())),
                kNormalAccumulationBlockSize,
                0,
                request.stream
            >>>(
                FaceNormalAccumulationView{
                    thrust::raw_pointer_cast(request.faces.data()),
                    thrust::raw_pointer_cast(request.workspace.accumulated_vertices_global.data()),
                    thrust::raw_pointer_cast(request.workspace.current_vertex_normal_sums.data()),
                    static_cast<int>(request.faces.size()),
                    request.first_new_vertex,
                    request.new_vertex_count,
                }
            );
            CUDA_CHECK(cudaGetLastError());
        }
    });

}

void materialize_device_face_normal_sums(DeviceNormalAccumulationRequest request)
{
    if (request.first_new_vertex < 0 || request.new_vertex_count < 0) {
        throw std::runtime_error("invalid current vertex range while materializing normals");
    }
    if (request.new_vertex_count == 0) {
        request.host_current_vertex_normal_sums.clear();
        return;
    }
    request.host_current_vertex_normal_sums.resize(static_cast<std::size_t>(request.new_vertex_count));
    {
        const Stopwatch stopwatch;
        cuda_detail::copy_device_to_host_sync(
            request.host_current_vertex_normal_sums.data(),
            thrust::raw_pointer_cast(request.workspace.current_vertex_normal_sums.data())
                + request.first_new_vertex,
            static_cast<std::size_t>(request.new_vertex_count),
            request.stream
        );
        if (request.timing != nullptr) {
            request.timing->add_stage(
                "stitch.incremental_normals.download_sums",
                stopwatch.elapsed_milliseconds()
            );
        }
    }
    {
        const Stopwatch stopwatch;
        for (int local_id = 0; local_id < request.new_vertex_count; ++local_id) {
            Vec3f& destination = request.normal_sums_global[
                static_cast<std::size_t>(request.first_new_vertex + local_id)
            ];
            const Vec3f source = request.host_current_vertex_normal_sums[
                static_cast<std::size_t>(local_id)
            ];
            destination.x += source.x;
            destination.y += source.y;
            destination.z += source.z;
        }
        if (request.timing != nullptr) {
            request.timing->add_stage(
                "stitch.incremental_normals.accumulate_host",
                stopwatch.elapsed_milliseconds()
            );
        }
    }
}

void accumulate_device_face_normal_sums(DeviceNormalAccumulationRequest request)
{
    launch_device_face_normal_sums(request);
    materialize_device_face_normal_sums(request);
}

void accumulate_host_face_normal_sums(
    const std::vector<Vec3f>& vertices_global,
    const std::vector<Triangle>& faces,
    std::vector<Vec3f>& normal_sums_global,
    PipelineTiming* timing
)
{
    accumulate_face_normal_sums_impl(vertices_global, faces, normal_sums_global, timing);
}

}  // namespace ri_meshing::cuda_pipeline
