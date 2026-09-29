#include "scan_internal.cuh"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ri_meshing::cuda_detail {
namespace {

constexpr float kPiF = 3.1415926535897932384626433832795F;

struct PointProjectionBuffers {
    const Vec3f* points{};
    int point_count{};
    int* pixel_last_index{};
};

struct MaterializedPixels {
    const Vec3f* points{};
    const int* pixel_last_index{};
    unsigned char* valid{};
    Vec3f* pixel_xyz{};
};

struct RangeComponentBuffers {
    const unsigned char* input{};
    const Vec3f* pixel_xyz{};
    unsigned char* output{};
};

static_assert(std::is_trivially_copyable_v<PointProjectionBuffers>);
static_assert(std::is_trivially_copyable_v<MaterializedPixels>);
static_assert(std::is_trivially_copyable_v<RangeComponentBuffers>);

__device__ float device_range(Vec3f point)
{
    return sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
}

__global__ void project_points_kernel(
    PointProjectionBuffers buffers,
    RiProjection<float> projection
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= buffers.point_count) {
        return;
    }

    const Vec3f point = buffers.points[idx];
    const float range = device_range(point);
    if (range <= 0.0F) {
        return;
    }

    const float xy = hypotf(point.x, point.y);
    const float phi = atan2f(point.z, xy);
    const float theta = atan2f(point.y, point.x);
    const int height = projection.dimensions.height;
    const int width = projection.dimensions.width;
    const int row = static_cast<int>(lroundf(
        (phi - projection.fov_min) / projection.fov_range * static_cast<float>(height)
    ));
    int col = static_cast<int>(lroundf((theta + kPiF) / (2.0F * kPiF) * static_cast<float>(width)));
    col = min(max(col, 0), width - 1);

    if (row >= 0 && row < height) {
        atomicMax(&buffers.pixel_last_index[row * width + col], idx);
    }
}

__global__ void materialize_pixels_kernel(
    MaterializedPixels buffers,
    int pixel_count
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }

    const int point_idx = buffers.pixel_last_index[pixel];
    buffers.valid[pixel] = point_idx >= 0;
    buffers.pixel_xyz[pixel] = point_idx >= 0 ? buffers.points[point_idx] : Vec3f{};
}

__global__ void filter_small_range_components_kernel(
    RangeComponentBuffers buffers,
    RiDimensions dimensions,
    float max_range_difference
)
{
    constexpr int kMaxRejectedComponentSize = 3;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = dimensions.height * dimensions.width;
    if (pixel >= pixel_count) {
        return;
    }
    if (buffers.input[pixel] == 0) {
        buffers.output[pixel] = 0;
        return;
    }

    // Stop as soon as a fourth reachable pixel is found. This is exact for
    // rejecting only components whose full connected size is at most three.
    int discovered[kMaxRejectedComponentSize + 1];
    discovered[0] = pixel;
    int discovered_count = 1;
    for (int current_idx = 0; current_idx < discovered_count; ++current_idx) {
        const int current = discovered[current_idx];
        const int row = current / dimensions.width;
        const int col = current - row * dimensions.width;
        const float current_range = device_range(buffers.pixel_xyz[current]);
        for (int dr = -1; dr <= 1; ++dr) {
            const int nr = row + dr;
            if (nr < 0 || nr >= dimensions.height) {
                continue;
            }
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) {
                    continue;
                }
                int nc = col + dc;
                if (nc < 0) {
                    nc += dimensions.width;
                } else if (nc >= dimensions.width) {
                    nc -= dimensions.width;
                }
                const int neighbor = nr * dimensions.width + nc;
                if (buffers.input[neighbor] == 0
                    || fabsf(current_range - device_range(buffers.pixel_xyz[neighbor]))
                        >= max_range_difference) {
                    continue;
                }
                bool already_discovered = false;
                for (int idx = 0; idx < discovered_count; ++idx) {
                    if (discovered[idx] == neighbor) {
                        already_discovered = true;
                        break;
                    }
                }
                if (already_discovered) {
                    continue;
                }
                if (discovered_count == kMaxRejectedComponentSize) {
                    buffers.output[pixel] = 1;
                    return;
                }
                discovered[discovered_count++] = neighbor;
            }
        }
    }
    buffers.output[pixel] = 0;
}

class ScanProjectionPass {
public:
    ScanProjectionPass(
        const std::vector<Vec3f>& points,
        const SparseMeshingConfig& config,
        CudaExecutionContext execution,
        ScanProjectionOptions options
    )
        : points_(points), config_(config), execution_(std::move(execution)), options_(options),
          height_(static_cast<int>(config.ri_height)), width_(static_cast<int>(config.ri_width)),
          pixel_count_(height_ * width_), point_count_(static_cast<int>(points.size()))
    {
    }

    CudaPixelScanPrepass run()
    {
        allocate_and_upload();
        project_points();
        materialize_pixels();
        filter_small_range_components();
        return CudaPixelScanPrepass{height_, width_, pixel_count_};
    }

private:
    static constexpr int kBlock = 256;

    static int grid_for(int count)
    {
        return (count + kBlock - 1) / kBlock;
    }

    void allocate_fill_storage()
    {
        CudaWorkspace& workspace = execution_.workspace;
        if (static_cast<int>(workspace.points.size()) < point_count_) {
            resize_with_headroom(workspace.points, static_cast<std::size_t>(point_count_));
        }
        if (static_cast<int>(workspace.pixel_last_index.size()) != pixel_count_) {
            resize_with_headroom(workspace.pixel_last_index, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.valid, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.clean_valid, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.meshing_valid, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.pixel_xyz, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.scan_ranges, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.filled_ranges, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.fill_scratch_ranges, static_cast<std::size_t>(pixel_count_));
            resize_with_headroom(workspace.filled_pixel_xyz, static_cast<std::size_t>(pixel_count_));
        }
        if (static_cast<int>(workspace.meshing_valid.size()) != pixel_count_) {
            resize_with_headroom(workspace.meshing_valid, static_cast<std::size_t>(pixel_count_));
        }
    }

    void allocate_projection_storage()
    {
        CudaWorkspace& workspace = execution_.workspace;
        workspace.points.assign(points_.begin(), points_.end());
        resize_with_headroom(workspace.pixel_last_index, static_cast<std::size_t>(pixel_count_));
        resize_with_headroom(workspace.valid, static_cast<std::size_t>(pixel_count_));
        resize_with_headroom(workspace.clean_valid, static_cast<std::size_t>(pixel_count_));
        resize_with_headroom(workspace.pixel_xyz, static_cast<std::size_t>(pixel_count_));
    }

    void allocate_and_upload()
    {
        timed_cuda_stage_void(execution_.timing, "scan.allocate_and_upload", [&] {
            CudaWorkspace& workspace = execution_.workspace;
            if (options_.allocate_fill_buffers) {
                allocate_fill_storage();
                if (point_count_ > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        thrust::raw_pointer_cast(workspace.points.data()), points_.data(),
                        static_cast<std::size_t>(point_count_) * sizeof(Vec3f),
                        cudaMemcpyHostToDevice, execution_.stream
                    ));
                }
            } else {
                allocate_projection_storage();
            }
            CUDA_CHECK(cudaMemsetAsync(
                thrust::raw_pointer_cast(workspace.pixel_last_index.data()), 0xFF,
                static_cast<std::size_t>(pixel_count_) * sizeof(int), execution_.stream
            ));
        });
    }

    void project_points()
    {
        timed_cuda_stage_void(execution_.timing, "scan.project_points", [&] {
            CudaWorkspace& workspace = execution_.workspace;
            const PointProjectionBuffers buffers{
                thrust::raw_pointer_cast(workspace.points.data()), point_count_,
                thrust::raw_pointer_cast(workspace.pixel_last_index.data()),
            };
            project_points_kernel<<<grid_for(point_count_), kBlock, 0, execution_.stream>>>(
                buffers, RiProjection<float>{
                    {height_, width_}, static_cast<float>(config_.vertical_fov.min_rad),
                    static_cast<float>(config_.vertical_fov.max_rad - config_.vertical_fov.min_rad),
                }
            );
        });
    }

    void materialize_pixels()
    {
        timed_cuda_stage_void(execution_.timing, "scan.materialize_pixels", [&] {
            CudaWorkspace& workspace = execution_.workspace;
            materialize_pixels_kernel<<<grid_for(pixel_count_), kBlock, 0, execution_.stream>>>(
                MaterializedPixels{
                    thrust::raw_pointer_cast(workspace.points.data()),
                    thrust::raw_pointer_cast(workspace.pixel_last_index.data()),
                    thrust::raw_pointer_cast(workspace.valid.data()),
                    thrust::raw_pointer_cast(workspace.pixel_xyz.data()),
                },
                pixel_count_
            );
        });
    }

    void filter_small_range_components()
    {
        timed_cuda_stage_void(execution_.timing, "scan.filter_small_range_components", [&] {
            CudaWorkspace& workspace = execution_.workspace;
            filter_small_range_components_kernel<<<grid_for(pixel_count_), kBlock, 0, execution_.stream>>>(
                RangeComponentBuffers{
                    thrust::raw_pointer_cast(workspace.valid.data()),
                    thrust::raw_pointer_cast(workspace.pixel_xyz.data()),
                    thrust::raw_pointer_cast(workspace.clean_valid.data()),
                },
                RiDimensions{height_, width_},
                static_cast<float>(config_.max_triangle_dist)
            );
            CUDA_CHECK(cudaGetLastError());
        });
    }

    const std::vector<Vec3f>& points_;
    const SparseMeshingConfig& config_;
    CudaExecutionContext execution_;
    ScanProjectionOptions options_;
    int height_{};
    int width_{};
    int pixel_count_{};
    int point_count_{};
};

}  // namespace

CudaPixelScanPrepass project_scan_to_ri_cuda(
    const std::vector<Vec3f>& points,
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    ScanProjectionOptions options
)
{
    return ScanProjectionPass(points, config, std::move(execution), options).run();
}

}  // namespace ri_meshing::cuda_detail
