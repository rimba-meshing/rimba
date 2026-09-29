#include "stitch_gpu_helpers.hpp"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"
#include "stitch_geometry.hpp"

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string_view>
#include <vector>

namespace ri_meshing::cuda_pipeline {
namespace {

class ProxyCudaEventTimer {
public:
    ProxyCudaEventTimer(bool enabled, cudaStream_t stream)
        : enabled_(enabled)
        , stream_(stream)
    {
        if (!enabled_) {
            return;
        }
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
        CUDA_CHECK(cudaEventRecord(start_, stream_));
    }

    ProxyCudaEventTimer(const ProxyCudaEventTimer&) = delete;
    ProxyCudaEventTimer& operator=(const ProxyCudaEventTimer&) = delete;

    ~ProxyCudaEventTimer()
    {
        if (start_ != nullptr) {
            static_cast<void>(cudaEventDestroy(start_));
        }
        if (stop_ != nullptr) {
            static_cast<void>(cudaEventDestroy(stop_));
        }
    }

    [[nodiscard]] double stop()
    {
        if (!enabled_ || stop_ == nullptr) {
            return 0.0;
        }
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        static_cast<void>(cudaEventDestroy(start_));
        static_cast<void>(cudaEventDestroy(stop_));
        start_ = nullptr;
        stop_ = nullptr;
        return static_cast<double>(milliseconds);
    }

private:
    bool enabled_{};
    cudaStream_t stream_{};
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

void print_proxy_timing(
    int frame_index,
    std::string_view phase,
    int items,
    double host_ms,
    double gpu_ms
)
{
    static std::mutex output_mutex;
    const std::lock_guard<std::mutex> lock(output_mutex);
    std::cerr << std::fixed << std::setprecision(6)
              << "stitch.proxy.kernel_timing"
              << " mode=polyline"
              << " frame=" << frame_index
              << " phase=" << phase
              << " items=" << items
              << " host_ms=" << host_ms
              << " gpu_ms=" << gpu_ms
              << '\n';
}

__device__ float debug_accumulated_depth_at(
    cuda_detail::AccumulatedDepthView depth,
    int row,
    int col,
    int height,
    int width
)
{
    if (depth.values == nullptr) {
        return __int_as_float(0x7f800000);
    }
    const int source_row = depth.bottom_up_normalized ? height - 1 - row : row;
    const float value = depth.values[source_row * width + col];
    if (!depth.bottom_up_normalized) {
        return value;
    }
    return value >= 0.999999F ? __int_as_float(0x7f800000) : value * 4096.0F;
}

__device__ std::uint64_t device_edge_key(int a, int b)
{
    const auto ua = static_cast<std::uint32_t>(a);
    const auto ub = static_cast<std::uint32_t>(b);
    const auto lo = ua < ub ? ua : ub;
    const auto hi = ua < ub ? ub : ua;
    return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

__device__ std::uint64_t pack_oriented_edge(int a, int b)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32U)
        | static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
}

__global__ void build_boundary_edge_records_kernel(
    const Triangle* faces,
    int face_count,
    std::uint64_t* keys,
    std::uint64_t* oriented_edges
)
{
    const int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= face_count) {
        return;
    }
    const Triangle face = faces[face_idx];
    const int edge_idx = face_idx * 3;
    keys[edge_idx] = device_edge_key(face.v0, face.v1);
    oriented_edges[edge_idx] = pack_oriented_edge(face.v0, face.v1);
    keys[edge_idx + 1] = device_edge_key(face.v1, face.v2);
    oriented_edges[edge_idx + 1] = pack_oriented_edge(face.v1, face.v2);
    keys[edge_idx + 2] = device_edge_key(face.v2, face.v0);
    oriented_edges[edge_idx + 2] = pack_oriented_edge(face.v2, face.v0);
}

__global__ void mark_unique_boundary_edges_kernel(
    const std::uint64_t* keys,
    int edge_count,
    int* flags
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_count) {
        return;
    }
    const std::uint64_t key = keys[idx];
    const bool different_from_previous = idx == 0 || keys[idx - 1] != key;
    const bool different_from_next = idx + 1 == edge_count || keys[idx + 1] != key;
    flags[idx] = different_from_previous && different_from_next ? 1 : 0;
}

__global__ void compact_unique_boundary_edges_kernel(
    const std::uint64_t* oriented_edges,
    const int* flags,
    const int* offsets,
    int edge_count,
    std::uint64_t* boundary_edges
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_count || flags[idx] == 0) {
        return;
    }
    boundary_edges[offsets[idx]] = oriented_edges[idx];
}

void batch_boundary_edges_sorted_cuda_impl(
    cuda_detail::CudaWorkspace& workspace,
    int old_face_count,
    int new_face_count,
    std::vector<std::array<int, 2>>& output
)
{
    output.clear();
    if (new_face_count <= 0) {
        return;
    }
    constexpr int block = 256;
    const auto grid_for = [](int count) { return (count + block - 1) / block; };
    const int edge_count = new_face_count * 3;
    cuda_detail::resize_with_headroom(workspace.edge_keys, static_cast<std::size_t>(edge_count));
    cuda_detail::resize_with_headroom(
        workspace.oriented_boundary_edges,
        static_cast<std::size_t>(edge_count)
    );
    cuda_detail::resize_with_headroom(workspace.flags, static_cast<std::size_t>(edge_count));
    cuda_detail::resize_with_headroom(workspace.offsets, static_cast<std::size_t>(edge_count));
    build_boundary_edge_records_kernel<<<grid_for(new_face_count), block>>>(
        thrust::raw_pointer_cast(workspace.accumulated_faces_global.data()) + old_face_count,
        new_face_count,
        thrust::raw_pointer_cast(workspace.edge_keys.data()),
        thrust::raw_pointer_cast(workspace.oriented_boundary_edges.data())
    );
    CUDA_CHECK(cudaGetLastError());
    thrust::sort_by_key(
        thrust::device,
        workspace.edge_keys.begin(),
        workspace.edge_keys.end(),
        workspace.oriented_boundary_edges.begin()
    );
    mark_unique_boundary_edges_kernel<<<grid_for(edge_count), block>>>(
        thrust::raw_pointer_cast(workspace.edge_keys.data()),
        edge_count,
        thrust::raw_pointer_cast(workspace.flags.data())
    );
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(thrust::device, workspace.flags.begin(), workspace.flags.end(), workspace.offsets.begin());

    int last_flag = 0;
    int last_offset = 0;
    CUDA_CHECK(cudaMemcpy(
        &last_flag,
        thrust::raw_pointer_cast(workspace.flags.data()) + edge_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        &last_offset,
        thrust::raw_pointer_cast(workspace.offsets.data()) + edge_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    const int boundary_count = last_offset + last_flag;
    cuda_detail::resize_with_headroom(
        workspace.boundary_edges,
        static_cast<std::size_t>(boundary_count)
    );
    if (boundary_count > 0) {
        compact_unique_boundary_edges_kernel<<<grid_for(edge_count), block>>>(
            thrust::raw_pointer_cast(workspace.oriented_boundary_edges.data()),
            thrust::raw_pointer_cast(workspace.flags.data()),
            thrust::raw_pointer_cast(workspace.offsets.data()),
            edge_count,
            thrust::raw_pointer_cast(workspace.boundary_edges.data())
        );
        CUDA_CHECK(cudaGetLastError());
    }

    workspace.host_boundary_edges.resize(static_cast<std::size_t>(boundary_count));
    cuda_detail::copy_device_to_host_sync(
        workspace.host_boundary_edges.data(),
        thrust::raw_pointer_cast(workspace.boundary_edges.data()),
        static_cast<std::size_t>(boundary_count)
    );
    output.reserve(workspace.host_boundary_edges.size());
    for (const std::uint64_t packed : workspace.host_boundary_edges) {
        output.push_back({
            static_cast<int>(packed >> 32U),
            static_cast<int>(packed & 0xFFFFFFFFULL),
        });
    }
}

__global__ void materialize_accumulated_depth_kernel(
    cuda_detail::AccumulatedDepthView depth,
    float* output,
    int height,
    int width
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = height * width;
    if (pixel >= pixel_count) {
        return;
    }
    const int row = pixel / width;
    const int col = pixel - row * width;
    output[pixel] = debug_accumulated_depth_at(depth, row, col, height, width);
}

struct ProxyDeviceMat4 {
    MeshScalar m[16];
};

struct ProxyRiShape {
    int height{};
    int width{};
};

struct ProxyProjectionInput {
    const Vec3f* vertices_global{};
    int vertex_count{};
    ProxyDeviceMat4 global_to_local;
    ProxyRiShape shape;
    MeshScalar fov_min{};
    MeshScalar fov_range{};
};

struct ProjectedProxyVertex {
    int row{};
    int col{};
    float range{};
};

struct ProxyDepthTarget {
    float* accumulated_depth{};
    ProxyRiShape shape;
    int radius{};
};

struct ProxyEdgePaintInput {
    ProxyDepthTarget target;
    ProxyProjectionInput projection;
    const std::uint64_t* edges{};
    int edge_count{};
};

ProxyDeviceMat4 proxy_to_device_mat(Mat4f mat)
{
    ProxyDeviceMat4 out{};
    for (int idx = 0; idx < 16; ++idx) {
        out.m[idx] = mat[idx];
    }
    return out;
}

std::uint64_t pack_proxy_edge(std::array<int, 2> edge)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(edge[0])) << 32U)
        | static_cast<std::uint64_t>(static_cast<std::uint32_t>(edge[1]));
}

void append_valid_proxy_edges(
    const std::vector<std::array<int, 2>>& edges,
    std::vector<std::uint64_t>& packed_edges
)
{
    for (const std::array<int, 2> edge : edges) {
        if (edge[0] >= 0 && edge[1] >= 0) {
            packed_edges.push_back(pack_proxy_edge(edge));
        }
    }
}

std::vector<std::uint64_t> pack_proxy_edges(const StitchProxyEdgeSet& proxy_edges)
{
    std::vector<std::uint64_t> packed_edges;
    packed_edges.reserve(
        proxy_edges.old_edges.size() + proxy_edges.new_edges.size() + proxy_edges.bridge_edges.size()
    );
    append_valid_proxy_edges(proxy_edges.old_edges, packed_edges);
    append_valid_proxy_edges(proxy_edges.new_edges, packed_edges);
    append_valid_proxy_edges(proxy_edges.bridge_edges, packed_edges);
    return packed_edges;
}

__device__ void proxy_atomic_min_positive_float(float* address, float value)
{
    if (!isfinite(value) || value <= 0.0F) {
        return;
    }
    atomicMin(reinterpret_cast<int*>(address), __float_as_int(value));
}

__device__ int proxy_unwrap_col_near(int reference_col, int col, int width)
{
    int out = col;
    const int half_width = width / 2;
    if (out - reference_col > half_width) {
        out -= width;
    } else if (reference_col - out > half_width) {
        out += width;
    }
    return out;
}

__device__ bool project_proxy_vertex(
    ProxyProjectionInput input,
    int vertex_id,
    ProjectedProxyVertex& output
)
{
    constexpr MeshScalar kPi = static_cast<MeshScalar>(3.14159265358979323846264338327950288);
    if (vertex_id < 0 || vertex_id >= input.vertex_count) {
        return false;
    }
    const Vec3f point_global = input.vertices_global[vertex_id];
    const MeshScalar tx = input.global_to_local.m[0] * point_global.x
        + input.global_to_local.m[1] * point_global.y
        + input.global_to_local.m[2] * point_global.z
        + input.global_to_local.m[3];
    const MeshScalar ty = input.global_to_local.m[4] * point_global.x
        + input.global_to_local.m[5] * point_global.y
        + input.global_to_local.m[6] * point_global.z
        + input.global_to_local.m[7];
    const MeshScalar tz = input.global_to_local.m[8] * point_global.x
        + input.global_to_local.m[9] * point_global.y
        + input.global_to_local.m[10] * point_global.z
        + input.global_to_local.m[11];
    const MeshScalar tw = input.global_to_local.m[12] * point_global.x
        + input.global_to_local.m[13] * point_global.y
        + input.global_to_local.m[14] * point_global.z
        + input.global_to_local.m[15];
    const MeshScalar inv_w = fabs(tw) > static_cast<MeshScalar>(1.0e-12)
        ? static_cast<MeshScalar>(1) / tw : static_cast<MeshScalar>(1);
    const float x = static_cast<float>(tx * inv_w);
    const float y = static_cast<float>(ty * inv_w);
    const float z = static_cast<float>(tz * inv_w);
    const MeshScalar xd = x;
    const MeshScalar yd = y;
    const MeshScalar zd = z;
    const MeshScalar rd = sqrt(xd * xd + yd * yd + zd * zd);
    if (!isfinite(rd) || rd <= static_cast<MeshScalar>(0)) {
        return false;
    }
    const MeshScalar xy = sqrt(xd * xd + yd * yd);
    const MeshScalar phi = atan2(zd, xy);
    const MeshScalar theta = atan2(yd, xd);
    output.row = static_cast<int>(llround(
        (phi - input.fov_min) / input.fov_range * static_cast<MeshScalar>(input.shape.height)
    ));
    output.col = static_cast<int>(llround(
        (theta + kPi) / (static_cast<MeshScalar>(2) * kPi) * static_cast<MeshScalar>(input.shape.width)
    ));
    output.col = max(0, min(input.shape.width - 1, output.col));
    if (output.row < 0 || output.row >= input.shape.height) {
        return false;
    }
    output.range = static_cast<float>(rd);
    return true;
}

__device__ void paint_proxy_depth_pixel(
    ProxyDepthTarget target,
    float range,
    int row,
    int col
)
{
    if (!isfinite(range) || range <= 0.0F) {
        return;
    }
    for (int dr = -target.radius; dr <= target.radius; ++dr) {
        const int rr = row + dr;
        if (rr < 0 || rr >= target.shape.height) {
            continue;
        }
        for (int dc = -target.radius; dc <= target.radius; ++dc) {
            int cc = col + dc;
            while (cc < 0) {
                cc += target.shape.width;
            }
            while (cc >= target.shape.width) {
                cc -= target.shape.width;
            }
            proxy_atomic_min_positive_float(
                &target.accumulated_depth[rr * target.shape.width + cc], range
            );
        }
    }
}

__global__ void paint_proxy_edges_depth_kernel(
    ProxyEdgePaintInput input
)
{
    const int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= input.edge_count) {
        return;
    }
    const std::uint64_t packed = input.edges[edge_idx];
    const int a_id = static_cast<int>(packed >> 32U);
    const int b_id = static_cast<int>(packed & 0xFFFFFFFFULL);
    ProjectedProxyVertex a;
    ProjectedProxyVertex b;
    if (!project_proxy_vertex(input.projection, a_id, a)
        || !project_proxy_vertex(input.projection, b_id, b)) {
        return;
    }
    const int width = input.target.shape.width;
    const int unwrapped_b_col = proxy_unwrap_col_near(a.col, b.col, width);
    const int drow = b.row - a.row;
    const int dcol = unwrapped_b_col - a.col;
    const int steps = max(abs(drow), abs(dcol));
    if (steps <= 0) {
        paint_proxy_depth_pixel(input.target, a.range, a.row, a.col);
        return;
    }
    for (int step = 0; step <= steps; ++step) {
        const MeshScalar t = static_cast<MeshScalar>(step) / static_cast<MeshScalar>(steps);
        const int row = static_cast<int>(llround(static_cast<MeshScalar>(a.row) + t * static_cast<MeshScalar>(drow)));
        int col = static_cast<int>(llround(static_cast<MeshScalar>(a.col) + t * static_cast<MeshScalar>(dcol)));
        while (col < 0) {
            col += width;
        }
        while (col >= width) {
            col -= width;
        }
        const float range = static_cast<float>(
            static_cast<MeshScalar>(a.range) + t * (static_cast<MeshScalar>(b.range) - static_cast<MeshScalar>(a.range))
        );
        paint_proxy_depth_pixel(input.target, range, row, col);
    }
}

struct ProxyRasterLaunch {
    ProxyRiShape shape;
    int pixel_count{};
    int vertex_count{};
    int edge_count{};
    int grid{};
    ProxyDeviceMat4 global_to_local;
    MeshScalar fov_min{};
    MeshScalar fov_range{};
    cudaStream_t stream{};
};

ProxyRasterLaunch make_proxy_raster_launch(
    const SparseMeshingConfig& config,
    const Mat4f& next_pose,
    int vertex_count,
    int edge_count,
    cudaStream_t stream
)
{
    constexpr int block = 256;
    const int height = static_cast<int>(config.ri_height);
    const int width = static_cast<int>(config.ri_width);
    return {
        {height, width},
        height * width,
        vertex_count,
        edge_count,
        (edge_count + block - 1) / block,
        proxy_to_device_mat(inverse_rigid_cpu(next_pose)),
        config.vertical_fov.min_rad,
        config.vertical_fov.max_rad - config.vertical_fov.min_rad,
        stream,
    };
}

void run_proxy_raster_stage(
    cuda_detail::CudaWorkspace& workspace,
    const std::vector<std::uint64_t>& packed_edges,
    const ProxyRasterLaunch& launch,
    ProxyPaintContext context
)
{
    constexpr int block = 256;
    constexpr int proxy_radius = 0;
    cuda_detail::timed_cuda_stage_void(context.timing, "stitch.proxy.raster_gpu", [&] {
        if (static_cast<int>(workspace.accumulated_depth.size()) != launch.pixel_count) {
            cuda_detail::resize_with_headroom(
                workspace.accumulated_depth,
                static_cast<std::size_t>(launch.pixel_count)
            );
            thrust::fill(
                thrust::cuda::par.on(launch.stream),
                workspace.accumulated_depth.begin(),
                workspace.accumulated_depth.end(),
                std::numeric_limits<float>::infinity()
            );
        }
        const Stopwatch upload_stopwatch;
        ProxyCudaEventTimer upload_timer(context.measure_kernel_timing, 0);
        workspace.edge_keys.assign(packed_edges.begin(), packed_edges.end());
        const double upload_ms = upload_stopwatch.elapsed_milliseconds();
        const double upload_gpu_ms = upload_timer.stop();
        ProxyCudaEventTimer paint_timer(context.measure_kernel_timing, launch.stream);
        paint_proxy_edges_depth_kernel<<<launch.grid, block, 0, launch.stream>>>(
            ProxyEdgePaintInput{
                ProxyDepthTarget{
                    thrust::raw_pointer_cast(workspace.accumulated_depth.data()),
                    launch.shape,
                    proxy_radius,
                },
                ProxyProjectionInput{
                    thrust::raw_pointer_cast(workspace.accumulated_vertices_global.data()),
                    launch.vertex_count,
                    launch.global_to_local,
                    launch.shape,
                    launch.fov_min,
                    launch.fov_range,
                },
                thrust::raw_pointer_cast(workspace.edge_keys.data()),
                launch.edge_count,
            }
        );
        CUDA_CHECK(cudaGetLastError());
        const double paint_ms = paint_timer.stop();
        if (context.measure_kernel_timing) {
            print_proxy_timing(
                context.frame_index,
                "assign_edge_buffer",
                launch.edge_count,
                upload_ms,
                upload_gpu_ms
            );
            print_proxy_timing(
                context.frame_index,
                "paint_proxy_edges",
                launch.edge_count,
                0.0,
                paint_ms
            );
        }
    });
}

void paint_stitch_proxy_edges_into_accumulated_cuda_impl(
    cuda_detail::CudaWorkspace& workspace,
    const StitchProxyEdgeSet& proxy_edges,
    const Mat4f& next_pose,
    ProxyPaintContext context
)
{
    const SparseMeshingConfig& config = context.config;
    const cudaStream_t stream = context.stream;
    if (proxy_edges.empty()) {
        return;
    }
    const int vertex_count = static_cast<int>(workspace.accumulated_vertices_global.size());
    if (vertex_count <= 0) {
        return;
    }
    const Stopwatch pack_stopwatch;
    const std::vector<std::uint64_t> packed_edges = pack_proxy_edges(proxy_edges);
    if (packed_edges.empty()) {
        return;
    }
    const double pack_ms = pack_stopwatch.elapsed_milliseconds();
    if (context.measure_kernel_timing) {
        print_proxy_timing(
            context.frame_index,
            "pack_edges",
            static_cast<int>(packed_edges.size()),
            pack_ms,
            0.0
        );
    }
    const int edge_count = static_cast<int>(packed_edges.size());
    const ProxyRasterLaunch launch = make_proxy_raster_launch(
        config, next_pose, vertex_count, edge_count, stream
    );
    run_proxy_raster_stage(workspace, packed_edges, launch, context);
}

}  // namespace

std::vector<std::array<int, 2>> batch_boundary_edges_sorted_cuda(
    cuda_detail::CudaWorkspace& workspace,
    int old_face_count,
    int new_face_count
)
{
    std::vector<std::array<int, 2>> output;
    batch_boundary_edges_sorted_cuda_impl(workspace, old_face_count, new_face_count, output);
    return output;
}

void batch_boundary_edges_sorted_cuda(
    cuda_detail::CudaWorkspace& workspace,
    int old_face_count,
    int new_face_count,
    std::vector<std::array<int, 2>>& output
)
{
    batch_boundary_edges_sorted_cuda_impl(workspace, old_face_count, new_face_count, output);
}

void paint_stitch_proxy_edges_into_accumulated_cuda(
    cuda_detail::CudaWorkspace& workspace,
    const StitchProxyEdgeSet& proxy_edges,
    const Mat4f& pose,
    ProxyPaintContext context
)
{
    paint_stitch_proxy_edges_into_accumulated_cuda_impl(workspace, proxy_edges, pose, context);
}

void materialize_accumulated_depth_cuda(
    cuda_detail::CudaWorkspace& workspace,
    cuda_detail::AccumulatedDepthView depth,
    int height,
    int width
)
{
    const int pixel_count = height * width;
    cuda_detail::resize_with_headroom(
        workspace.accumulated_depth,
        static_cast<std::size_t>(pixel_count)
    );
    constexpr int block = 256;
    const int grid = (pixel_count + block - 1) / block;
    materialize_accumulated_depth_kernel<<<grid, block>>>(
        depth, thrust::raw_pointer_cast(workspace.accumulated_depth.data()), height, width
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ri_meshing::cuda_pipeline
