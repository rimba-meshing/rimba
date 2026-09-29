#include "scan_internal.cuh"

#include "cuda_utils.hpp"
#include "device_vector_memory.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace ri_meshing::cuda_detail {
namespace {

constexpr float kPiF = 3.1415926535897932384626433832795F;
constexpr int kFilledPixelFastIterativeFillIterations = 5;

struct RangeFillBuffers {
    const float* input{};
    float* output{};
    // A one means that the range has a current-scan ancestor. Old-only ranges
    // are provisional support: they may participate in interpolation and can
    // later be upgraded when a current-scan path becomes coherent.
    const unsigned char* input_anchor{};
    unsigned char* output_anchor{};
};

struct RangeFillParams {
    RiDimensions dimensions;
    int row_radius{};
    int column_radius{};
    float discontinuity_threshold{};
    float threshold_scale_limit{};
};

struct ProjectedVertexSupportBuffers {
    const int* rows{};
    const int* cols{};
    const float* ranges{};
    float* support_ranges{};
};

struct ProjectedVertexSupportParams {
    int vertex_count{};
    int width{};
};

struct CurrentAnchorBuffers {
    const unsigned char* current_valid{};
    unsigned char* anchors{};
};

struct FilledXyzBuffers {
    const float* ranges{};
    const unsigned char* original_valid{};
    const Vec3f* original_xyz{};
    const unsigned char* anchor{};
    unsigned char* filled_valid{};
    Vec3f* filled_xyz{};
};

static_assert(std::is_trivially_copyable_v<RangeFillBuffers>);
static_assert(std::is_trivially_copyable_v<RangeFillParams>);
static_assert(std::is_trivially_copyable_v<ProjectedVertexSupportBuffers>);
static_assert(std::is_trivially_copyable_v<ProjectedVertexSupportParams>);
static_assert(std::is_trivially_copyable_v<CurrentAnchorBuffers>);
static_assert(std::is_trivially_copyable_v<FilledXyzBuffers>);

__device__ float device_range_f(Vec3f point)
{
    return sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
}

__global__ void materialize_scan_ranges_kernel(
    const unsigned char* valid,
    const Vec3f* pixel_xyz,
    float* scan_ranges,
    int pixel_count
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    scan_ranges[pixel] = valid[pixel] != 0 ? device_range_f(pixel_xyz[pixel]) : __int_as_float(0x7f800000);
}

__global__ void mask_ranges_for_meshing_kernel(
    const float* scan_ranges,
    const unsigned char* valid,
    float* meshing_ranges,
    int pixel_count
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    meshing_ranges[pixel] = valid[pixel] != 0 ? scan_ranges[pixel] : __int_as_float(0x7f800000);
}

__device__ bool range_pixel_valid(float range)
{
    return isfinite(range) && range != 0.0F;
}

__device__ float accumulated_support_range_at(
    AccumulatedDepthView depth,
    int row,
    int col,
    RiDimensions dimensions
)
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

__global__ void materialize_accumulated_support_kernel(
    AccumulatedDepthView depth,
    float* support_ranges,
    RiDimensions dimensions
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = dimensions.height * dimensions.width;
    if (pixel >= pixel_count) {
        return;
    }
    const int row = pixel / dimensions.width;
    const int col = pixel - row * dimensions.width;
    support_ranges[pixel] = accumulated_support_range_at(depth, row, col, dimensions);
}

__global__ void initialize_current_anchors_kernel(CurrentAnchorBuffers buffers, int pixel_count)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel < pixel_count) {
        buffers.anchors[pixel] = buffers.current_valid[pixel] != 0;
    }
}

__global__ void stamp_projected_vertex_support_kernel(
    ProjectedVertexSupportBuffers buffers,
    ProjectedVertexSupportParams params
)
{
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= params.vertex_count) {
        return;
    }
    const int row = buffers.rows[vertex];
    const int col = buffers.cols[vertex];
    const float range = buffers.ranges[vertex];
    if (row < 0 || col < 0 || !range_pixel_valid(range)) {
        return;
    }
    atomicMin(
        reinterpret_cast<int*>(&buffers.support_ranges[row * params.width + col]),
        __float_as_int(range)
    );
}

__global__ void merge_accumulated_support_kernel(
    float* ranges,
    const float* support_ranges,
    unsigned char* anchors,
    int pixel_count
)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    if (range_pixel_valid(ranges[pixel])) {
        // The range came from the accepted current scan before support merge.
        anchors[pixel] = 1;
        return;
    }
    if (range_pixel_valid(support_ranges[pixel])) {
        ranges[pixel] = support_ranges[pixel];
        anchors[pixel] = 0;
        return;
    }
    anchors[pixel] = 0;
}

__device__ Vec3f ri_pixel_to_xyz_f(
    int row,
    int col,
    float point_range,
    RiProjection<float> projection
)
{
    const float phi = projection.fov_min
        + (static_cast<float>(row) / static_cast<float>(projection.dimensions.height)) * projection.fov_range;
    const float theta = (static_cast<float>(col) / static_cast<float>(projection.dimensions.width))
        * 2.0F * kPiF - kPiF;
    const float xy = point_range * cosf(phi);
    return Vec3f{xy * cosf(theta), xy * sinf(theta), point_range * sinf(phi)};
}

__global__ void fill_irregular_horizontal_kernel(
    RangeFillBuffers buffers,
    RangeFillParams params
)
{
    const float* ranges_in = buffers.input;
    float* ranges_out = buffers.output;
    const int height = params.dimensions.height;
    const int width = params.dimensions.width;
    const int forgiveness_j = params.column_radius;
    const float range_discontinuity_threshold = params.discontinuity_threshold;
    const float range_threshold_scale_limit = params.threshold_scale_limit;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = height * width;
    if (pixel >= pixel_count) {
        return;
    }
    const float current = ranges_in[pixel];
    const bool provisional_old = range_pixel_valid(current)
        && buffers.input_anchor != nullptr && buffers.input_anchor[pixel] == 0;
    if (range_pixel_valid(current) && !provisional_old) {
        ranges_out[pixel] = current;
        if (buffers.output_anchor != nullptr) {
            buffers.output_anchor[pixel] = buffers.input_anchor[pixel];
        }
        return;
    }

    const int row = pixel / width;
    const int col = pixel - row * width;
    int left_col = -1;
    int right_col = -1;
    float left_value = 0.0F;
    float right_value = 0.0F;
    for (int offset = 1; offset <= forgiveness_j; ++offset) {
        // The angular fill radius is strictly smaller than one RI row width,
        // so one conditional wrap is equivalent to modulo and avoids an
        // integer division in this hot loop.
        int candidate_col = col - offset;
        if (candidate_col < 0) {
            candidate_col += width;
        }
        const float value = ranges_in[row * width + candidate_col];
        if (range_pixel_valid(value)) {
            left_col = candidate_col;
            left_value = value;
            break;
        }
    }
    for (int offset = 1; offset <= forgiveness_j; ++offset) {
        int candidate_col = col + offset;
        if (candidate_col >= width) {
            candidate_col -= width;
        }
        const float value = ranges_in[row * width + candidate_col];
        if (range_pixel_valid(value)) {
            right_col = candidate_col;
            right_value = value;
            break;
        }
    }
    if (left_col < 0 || right_col < 0) {
        if (provisional_old) {
            ranges_out[pixel] = current;
            buffers.output_anchor[pixel] = 0;
        } else {
            ranges_out[pixel] = __int_as_float(0x7f800000);
            if (buffers.output_anchor != nullptr) {
                buffers.output_anchor[pixel] = 0;
            }
        }
        return;
    }
    int dist_left = col - left_col;
    if (dist_left < 0) {
        dist_left = -dist_left;
    }
    if (width - dist_left < dist_left) {
        dist_left = width - dist_left;
    }
    int dist_right = col - right_col;
    if (dist_right < 0) {
        dist_right = -dist_right;
    }
    if (width - dist_right < dist_right) {
        dist_right = width - dist_right;
    }
    const float total = static_cast<float>(dist_left + dist_right);
    const float threshold_scale = fminf(total, fmaxf(1.0F, range_threshold_scale_limit));
    const float threshold = range_discontinuity_threshold * threshold_scale;
    if (fabsf(left_value - right_value) > threshold) {
        if (provisional_old) {
            ranges_out[pixel] = current;
            buffers.output_anchor[pixel] = 0;
        } else {
            ranges_out[pixel] = __int_as_float(0x7f800000);
            if (buffers.output_anchor != nullptr) {
                buffers.output_anchor[pixel] = 0;
            }
        }
        return;
    }
    const bool candidate_current = buffers.input_anchor != nullptr
        && (buffers.input_anchor[row * width + left_col] != 0
            || buffers.input_anchor[row * width + right_col] != 0);
    if (provisional_old && !candidate_current) {
        ranges_out[pixel] = current;
        buffers.output_anchor[pixel] = 0;
    } else {
        ranges_out[pixel] = (static_cast<float>(dist_right) * left_value
            + static_cast<float>(dist_left) * right_value) / total;
        if (buffers.output_anchor != nullptr) {
            buffers.output_anchor[pixel] = static_cast<unsigned char>(candidate_current);
        }
    }
}

__global__ void fill_irregular_vertical_kernel(
    RangeFillBuffers buffers,
    RangeFillParams params
)
{
    const float* ranges_in = buffers.input;
    float* ranges_out = buffers.output;
    const int height = params.dimensions.height;
    const int width = params.dimensions.width;
    const int forgiveness_i = params.row_radius;
    const float range_discontinuity_threshold = params.discontinuity_threshold;
    const float range_threshold_scale_limit = params.threshold_scale_limit;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = height * width;
    if (pixel >= pixel_count) {
        return;
    }
    const float current = ranges_in[pixel];
    const bool provisional_old = range_pixel_valid(current)
        && buffers.input_anchor != nullptr && buffers.input_anchor[pixel] == 0;
    if (range_pixel_valid(current) && !provisional_old) {
        ranges_out[pixel] = current;
        if (buffers.output_anchor != nullptr) {
            buffers.output_anchor[pixel] = buffers.input_anchor[pixel];
        }
        return;
    }

    const int row = pixel / width;
    const int col = pixel - row * width;
    int up_row = -1;
    int down_row = -1;
    float up_value = 0.0F;
    float down_value = 0.0F;
    for (int offset = 1; offset <= forgiveness_i; ++offset) {
        const int candidate_row = row - offset;
        if (candidate_row < 0) {
            break;
        }
        const float value = ranges_in[candidate_row * width + col];
        if (range_pixel_valid(value)) {
            up_row = candidate_row;
            up_value = value;
            break;
        }
    }
    for (int offset = 1; offset <= forgiveness_i; ++offset) {
        const int candidate_row = row + offset;
        if (candidate_row >= height) {
            break;
        }
        const float value = ranges_in[candidate_row * width + col];
        if (range_pixel_valid(value)) {
            down_row = candidate_row;
            down_value = value;
            break;
        }
    }
    if (up_row < 0 || down_row < 0) {
        if (provisional_old) {
            ranges_out[pixel] = current;
            buffers.output_anchor[pixel] = 0;
        } else {
            ranges_out[pixel] = __int_as_float(0x7f800000);
            if (buffers.output_anchor != nullptr) {
                buffers.output_anchor[pixel] = 0;
            }
        }
        return;
    }
    const float dist_up = static_cast<float>(row - up_row);
    const float dist_down = static_cast<float>(down_row - row);
    const float total = dist_up + dist_down;
    const float threshold_scale = fminf(total, fmaxf(1.0F, range_threshold_scale_limit));
    const float threshold = range_discontinuity_threshold * threshold_scale;
    if (fabsf(up_value - down_value) > threshold) {
        if (provisional_old) {
            ranges_out[pixel] = current;
            buffers.output_anchor[pixel] = 0;
        } else {
            ranges_out[pixel] = __int_as_float(0x7f800000);
            if (buffers.output_anchor != nullptr) {
                buffers.output_anchor[pixel] = 0;
            }
        }
        return;
    }
    const bool candidate_current = buffers.input_anchor != nullptr
        && (buffers.input_anchor[up_row * width + col] != 0
            || buffers.input_anchor[down_row * width + col] != 0);
    if (provisional_old && !candidate_current) {
        ranges_out[pixel] = current;
        buffers.output_anchor[pixel] = 0;
    } else {
        ranges_out[pixel] = (dist_down * up_value + dist_up * down_value) / total;
        if (buffers.output_anchor != nullptr) {
            buffers.output_anchor[pixel] = static_cast<unsigned char>(candidate_current);
        }
    }
}

__global__ void ranges_to_xyz_kernel(
    FilledXyzBuffers buffers,
    RiProjection<float> projection
)
{
    const int height = projection.dimensions.height;
    const int width = projection.dimensions.width;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = height * width;
    if (pixel >= pixel_count) {
        return;
    }
    const float range = buffers.ranges[pixel];
    if (!range_pixel_valid(range)) {
        buffers.filled_valid[pixel] = 0;
        buffers.filled_xyz[pixel] = Vec3f{};
        return;
    }
    if (buffers.anchor != nullptr && buffers.anchor[pixel] == 0) {
        buffers.filled_valid[pixel] = 0;
        buffers.filled_xyz[pixel] = Vec3f{};
        return;
    }
    if (buffers.original_valid[pixel] != 0) {
        buffers.filled_valid[pixel] = 1;
        buffers.filled_xyz[pixel] = buffers.original_xyz[pixel];
        return;
    }
    const int row = pixel / width;
    const int col = pixel - row * width;
    buffers.filled_valid[pixel] = 1;
    buffers.filled_xyz[pixel] = ri_pixel_to_xyz_f(row, col, range, projection);
}

class RangeFillPass {
public:
    RangeFillPass(
        const SparseMeshingConfig& config,
        CudaExecutionContext execution,
        CudaPixelScanPrepass prepass,
        ScanBuildOptions options
    )
        : config_(config), execution_(std::move(execution)), prepass_(prepass), options_(options),
          fov_min_(static_cast<float>(config.vertical_fov.min_rad)),
          fov_range_(static_cast<float>(
              config.vertical_fov.max_rad - config.vertical_fov.min_rad
          ))
    {
    }

    void run()
    {
        materialize_ranges();
        if (tracks_anchors()) {
            prepare_fill_anchors();
        }
        fill_irregular_range_image();
        if (options_.accumulated_fill_support != nullptr) {
            prepare_accumulated_support();
            merge_accumulated_support();
            fill_irregular_range_image();
        }
        convert_ranges_to_xyz();
    }

    // The pre-scan worker has already materialized and filled the current
    // scan. Once the exact accumulated depth is ready, only add old support
    // and repeat the normal fill pass; do not redo projection or the first
    // current-scan fill.
    void run_deferred_accumulated_support()
    {
        if (options_.accumulated_fill_support == nullptr) {
            return;
        }
        prepare_accumulated_support();
        merge_accumulated_support();
        fill_irregular_range_image();
        convert_ranges_to_xyz();
    }

private:
    static constexpr int kBlock = 256;

    int grid_for_pixels() const
    {
        return (prepass_.pixel_count + kBlock - 1) / kBlock;
    }

    RangeFillBuffers fill_buffers(bool scratch_is_output)
    {
        CudaWorkspace& workspace = execution_.workspace;
        const bool track_anchors = tracks_anchors();
        return scratch_is_output
            ? RangeFillBuffers{
                thrust::raw_pointer_cast(workspace.filled_ranges.data()),
                thrust::raw_pointer_cast(workspace.fill_scratch_ranges.data()),
                track_anchors ? thrust::raw_pointer_cast(workspace.fill_anchor.data()) : nullptr,
                track_anchors ? thrust::raw_pointer_cast(workspace.fill_scratch_anchor.data()) : nullptr,
            }
            : RangeFillBuffers{
                thrust::raw_pointer_cast(workspace.fill_scratch_ranges.data()),
                thrust::raw_pointer_cast(workspace.filled_ranges.data()),
                track_anchors ? thrust::raw_pointer_cast(workspace.fill_scratch_anchor.data()) : nullptr,
                track_anchors ? thrust::raw_pointer_cast(workspace.fill_anchor.data()) : nullptr,
            };
    }

    bool tracks_anchors() const
    {
        return options_.accumulated_fill_support != nullptr || options_.track_fill_anchors;
    }

    RangeFillParams fill_params(int row_radius, int column_radius, float scale_limit) const
    {
        return RangeFillParams{
            {prepass_.height, prepass_.width}, row_radius, column_radius,
            static_cast<float>(config_.max_triangle_dist), scale_limit,
        };
    }

    void materialize_ranges()
    {
        timed_cuda_stage_void(execution_.timing, "scan.materialize_ranges", [&] {
            CudaWorkspace& workspace = execution_.workspace;
            materialize_scan_ranges_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                thrust::raw_pointer_cast(workspace.valid.data()),
                thrust::raw_pointer_cast(workspace.pixel_xyz.data()),
                thrust::raw_pointer_cast(workspace.scan_ranges.data()),
                prepass_.pixel_count
            );
            CUDA_CHECK(cudaGetLastError());
        });

        CudaWorkspace& workspace = execution_.workspace;
        timed_cuda_stage_void(execution_.timing, "scan.prepare_meshing_valid", [&] {
            copy_device_to_device_async(
                thrust::raw_pointer_cast(workspace.meshing_valid.data()),
                thrust::raw_pointer_cast(workspace.clean_valid.data()),
                static_cast<std::size_t>(prepass_.pixel_count),
                execution_.stream
            );
        });

        timed_cuda_stage_void(execution_.timing, "scan.prepare_meshing_ranges", [&] {
            mask_ranges_for_meshing_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                thrust::raw_pointer_cast(workspace.scan_ranges.data()),
                thrust::raw_pointer_cast(workspace.meshing_valid.data()),
                thrust::raw_pointer_cast(workspace.filled_ranges.data()),
                prepass_.pixel_count
            );
            CUDA_CHECK(cudaGetLastError());
        });
    }

    void run_fill_iterations(int iterations, RangeFillParams params)
    {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            fill_irregular_horizontal_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                fill_buffers(true), params
            );
            CUDA_CHECK(cudaGetLastError());
            fill_irregular_vertical_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                fill_buffers(false), params
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }

    void prepare_accumulated_support()
    {
        if (options_.accumulated_fill_support == nullptr) {
            return;
        }
        CudaWorkspace& workspace = execution_.workspace;
        resize_with_headroom(workspace.fill_support_ranges, static_cast<std::size_t>(prepass_.pixel_count));
        timed_cuda_stage_void(execution_.timing, "scan.prepare_accumulated_fill_support", [&] {
            materialize_accumulated_support_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                *options_.accumulated_fill_support,
                thrust::raw_pointer_cast(workspace.fill_support_ranges.data()),
                RiDimensions{prepass_.height, prepass_.width}
            );
            CUDA_CHECK(cudaGetLastError());
            ProjectedVertexSupportView projected = options_.projected_vertex_support;
            const bool has_external_projection = projected.rows != nullptr
                && projected.cols != nullptr
                && projected.ranges != nullptr;
            if (!has_external_projection) {
                projected = ProjectedVertexSupportView{
                    thrust::raw_pointer_cast(workspace.projected_rows.data()),
                    thrust::raw_pointer_cast(workspace.projected_cols.data()),
                    thrust::raw_pointer_cast(workspace.projected_ranges.data()),
                    options_.old_vertex_count,
                };
            }
            const bool projection_buffers_ready = projected.vertex_count > 0
                && projected.rows != nullptr
                && projected.cols != nullptr
                && projected.ranges != nullptr
                && (
                    has_external_projection
                    || (
                        static_cast<int>(workspace.projected_rows.size()) >= projected.vertex_count
                        && static_cast<int>(workspace.projected_cols.size()) >= projected.vertex_count
                        && static_cast<int>(workspace.projected_ranges.size()) >= projected.vertex_count
                    )
                );
            if (projection_buffers_ready) {
                stamp_projected_vertex_support_kernel<<<
                    (projected.vertex_count + kBlock - 1) / kBlock, kBlock, 0, execution_.stream
                >>>(
                    ProjectedVertexSupportBuffers{
                        projected.rows,
                        projected.cols,
                        projected.ranges,
                        thrust::raw_pointer_cast(workspace.fill_support_ranges.data()),
                    },
                    ProjectedVertexSupportParams{projected.vertex_count, prepass_.width}
                );
                CUDA_CHECK(cudaGetLastError());
            }
        });
    }

    void prepare_fill_anchors()
    {
        CudaWorkspace& workspace = execution_.workspace;
        resize_with_headroom(workspace.fill_anchor, static_cast<std::size_t>(prepass_.pixel_count));
        resize_with_headroom(workspace.fill_scratch_anchor, static_cast<std::size_t>(prepass_.pixel_count));
        initialize_current_anchors_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
            CurrentAnchorBuffers{
                thrust::raw_pointer_cast(workspace.meshing_valid.data()),
                thrust::raw_pointer_cast(workspace.fill_anchor.data()),
            },
            prepass_.pixel_count
        );
        CUDA_CHECK(cudaGetLastError());
    }

    void merge_accumulated_support()
    {
        CudaWorkspace& workspace = execution_.workspace;
        timed_cuda_stage_void(execution_.timing, "scan.merge_accumulated_fill_support", [&] {
            merge_accumulated_support_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                thrust::raw_pointer_cast(workspace.filled_ranges.data()),
                thrust::raw_pointer_cast(workspace.fill_support_ranges.data()),
                thrust::raw_pointer_cast(workspace.fill_anchor.data()),
                prepass_.pixel_count
            );
            CUDA_CHECK(cudaGetLastError());
        });
    }

    void fill_irregular_range_image()
    {
        timed_cuda_stage_void(execution_.timing, "scan.fill_irregular_ri", [&] {
            const int row_radius = static_cast<int>(
                (kPiF / 180.0F) / fov_range_ * static_cast<float>(prepass_.height)
            );
            const int column_radius = static_cast<int>(
                (kPiF / 180.0F) / (2.0F * kPiF) * static_cast<float>(prepass_.width)
            );
            run_fill_iterations(
                kFilledPixelFastIterativeFillIterations,
                fill_params(row_radius, column_radius, 1.0F)
            );
        });
    }

    void convert_ranges_to_xyz()
    {
        timed_cuda_stage_void(execution_.timing, "scan.ranges_to_xyz", [&] {
            CudaWorkspace& workspace = execution_.workspace;
            ranges_to_xyz_kernel<<<grid_for_pixels(), kBlock, 0, execution_.stream>>>(
                FilledXyzBuffers{
                    thrust::raw_pointer_cast(workspace.filled_ranges.data()),
                    thrust::raw_pointer_cast(workspace.meshing_valid.data()),
                    thrust::raw_pointer_cast(workspace.pixel_xyz.data()),
                    !tracks_anchors()
                        ? nullptr : thrust::raw_pointer_cast(workspace.fill_anchor.data()),
                    thrust::raw_pointer_cast(workspace.valid.data()),
                    thrust::raw_pointer_cast(workspace.filled_pixel_xyz.data()),
                },
                RiProjection<float>{{prepass_.height, prepass_.width}, fov_min_, fov_range_}
            );
            CUDA_CHECK(cudaGetLastError());
        });
    }

    const SparseMeshingConfig& config_;
    CudaExecutionContext execution_;
    CudaPixelScanPrepass prepass_;
    ScanBuildOptions options_;
    float fov_min_{};
    float fov_range_{};
};

}  // namespace

void fill_scan_range_image_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    CudaPixelScanPrepass prepass,
    const ScanBuildOptions& options
)
{
    RangeFillPass(config, std::move(execution), prepass, options).run();
}

void apply_accumulated_fill_support_cuda(
    const SparseMeshingConfig& config,
    CudaExecutionContext execution,
    CudaPixelScanPrepass prepass,
    const ScanBuildOptions& options
)
{
    RangeFillPass(config, std::move(execution), prepass, options)
        .run_deferred_accumulated_support();
}

}  // namespace ri_meshing::cuda_detail
