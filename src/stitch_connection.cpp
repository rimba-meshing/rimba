#include "stitch_connection.hpp"

#include "ri_meshing/timing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ri_meshing::stitch_detail {
namespace {

constexpr MeshScalar kPi = static_cast<MeshScalar>(3.14159265358979323846264338327950288);

}  // namespace

StitchConnectionOracle::StitchConnectionOracle(
    const StitchConnectionOracleRequest& request
)
    : mask_(request.mask), config_(request.config), vertices_(request.vertices),
      timing_(request.timing),
      pixel_count_(mask_.height > 0 && mask_.width > 0
          ? static_cast<std::size_t>(mask_.height) * static_cast<std::size_t>(mask_.width)
          : 0),
      mask_pixels_available_(mask_.pixels.size() == pixel_count_),
      old_support_available_(mask_.old_support.size() == pixel_count_),
      new_support_available_(mask_.new_support.size() == pixel_count_),
      projected_pixels_(request.vertices.size()),
      nearest_empty_by_vertex_(request.vertices.size(), -1),
      vertex_components_(request.vertices.size())
{
    Stopwatch projection_stopwatch;
    const auto project_paths = [&](const std::vector<BoundaryPath>& paths) {
        for (const BoundaryPath& path : paths) {
            for (const int vertex : path.ids) {
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertices_.size()) {
                    continue;
                }
                projected_pixels_[static_cast<std::size_t>(vertex)] =
                    project(vertices_[static_cast<std::size_t>(vertex)]);
            }
        }
    };
    project_paths(request.old_paths);
    project_paths(request.new_paths);
    if (timing_ != nullptr) {
        timing_->add_stage(
            "stitch.core.connection_oracle.project_vertices",
            projection_stopwatch.elapsed_milliseconds()
        );
    }
    Stopwatch stopwatch;
    build_components();
    if (timing_ != nullptr) {
        timing_->add_stage("stitch.core.connection_oracle.components", stopwatch.elapsed_milliseconds());
    }
    build_linear_support_offsets();
    const Stopwatch cache_stopwatch;
    prepare_nearest_empty_cache(request.old_paths, request.new_paths);
    const double cache_milliseconds = cache_stopwatch.elapsed_milliseconds();
    double old_support_milliseconds = 0.0;
    double new_support_milliseconds = 0.0;
#if defined(RI_MESHING_HAS_OPENMP)
    const bool enough_support_work = request.old_paths.size() + request.new_paths.size() > 8;
    #pragma omp parallel sections if(enough_support_work)
    {
        #pragma omp section
        {
            const Stopwatch support_stopwatch;
            add_boundary_support(request.old_paths);
            old_support_milliseconds += support_stopwatch.elapsed_milliseconds();
        }
        #pragma omp section
        {
            const Stopwatch support_stopwatch;
            add_boundary_support(request.new_paths);
            new_support_milliseconds += support_stopwatch.elapsed_milliseconds();
        }
    }
#else
    {
        const Stopwatch support_stopwatch;
        add_boundary_support(request.old_paths);
        old_support_milliseconds += support_stopwatch.elapsed_milliseconds();
    }
    {
        const Stopwatch support_stopwatch;
        add_boundary_support(request.new_paths);
        new_support_milliseconds += support_stopwatch.elapsed_milliseconds();
    }
#endif
    if (timing_ != nullptr) {
        timing_->add_stage(
            "stitch.core.connection_oracle.nearest_empty_cache", cache_milliseconds
        );
        timing_->add_stage(
            "stitch.core.connection_oracle.old_support", old_support_milliseconds
        );
        timing_->add_stage(
            "stitch.core.connection_oracle.new_support", new_support_milliseconds
        );
    }
}

std::array<int, 2> StitchConnectionOracle::projected_pixel(int vertex) const
{
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= projected_pixels_.size()) {
        return {-1, -1};
    }
    Pixel pixel = projected_pixels_[static_cast<std::size_t>(vertex)];
    if (pixel.row < 0 && pixel.col < 0) {
        pixel = project(vertices_[static_cast<std::size_t>(vertex)]);
    }
    return {pixel.row, pixel.col};
}

void StitchConnectionOracle::inherit_component_support(int vertex, int component)
{
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertices_.size()) {
        return;
    }
    add_vertex_component(
        vertex,
        component,
        nearest_component_pixel(projected_pixel_for_vertex(vertex), component)
    );
}

StitchConnectionOracle::Pixel StitchConnectionOracle::project(Vec3f point) const
{
    const MeshScalar x = point.x;
    const MeshScalar y = point.y;
    const MeshScalar z = point.z;
    const MeshScalar elevation = std::atan2(z, std::hypot(x, y));
    const MeshScalar azimuth = std::atan2(y, x);
    const MeshScalar fov_min = config_.vertical_fov.min_rad;
    const MeshScalar fov_range = config_.vertical_fov.max_rad - fov_min;
    const int row = static_cast<int>(std::round(
        (elevation - fov_min) / fov_range * static_cast<MeshScalar>(mask_.height)
    ));
    const int raw_col = static_cast<int>(std::round(
        (azimuth + kPi) / (static_cast<MeshScalar>(2) * kPi) * static_cast<MeshScalar>(mask_.width)
    ));
    // Match the dense rasterizer and boundary label projection: vertex
    // projection clamps the +pi endpoint to the last valid column. Horizontal
    // wrapping is reserved for neighbour walks and interpolated segments.
    const int col = std::clamp(raw_col, 0, mask_.width - 1);
    return Pixel{row, col};
}

StitchConnectionOracle::Pixel StitchConnectionOracle::projected_pixel_for_vertex(int vertex) const
{
    const ComponentSupport& support = vertex_components_[static_cast<std::size_t>(vertex)];
    if (support.component >= 0) {
        return support.vertex_pixel;
    }
    Pixel pixel = projected_pixels_[static_cast<std::size_t>(vertex)];
    if (pixel.row < 0 && pixel.col < 0) {
        pixel = project(vertices_[static_cast<std::size_t>(vertex)]);
    }
    return pixel;
}

int StitchConnectionOracle::wrap_column(int col) const
{
    if (mask_.width <= 0) {
        return -1;
    }
    col %= mask_.width;
    return col < 0 ? col + mask_.width : col;
}

bool StitchConnectionOracle::valid_pixel(Pixel pixel) const
{
    return pixel.row >= 0 && pixel.row < mask_.height
        && pixel.col >= 0 && pixel.col < mask_.width;
}

int StitchConnectionOracle::pixel_chebyshev_distance(Pixel first, Pixel second) const
{
    if (!valid_pixel(first) || !valid_pixel(second)) {
        return std::numeric_limits<int>::max();
    }
    int delta_col = second.col - first.col;
    if (delta_col > mask_.width / 2) {
        delta_col -= mask_.width;
    } else if (delta_col < -mask_.width / 2) {
        delta_col += mask_.width;
    }
    return std::max(std::abs(second.row - first.row), std::abs(delta_col));
}

int StitchConnectionOracle::pixel_distance_squared(Pixel first, Pixel second) const
{
    if (!valid_pixel(first) || !valid_pixel(second)) {
        return std::numeric_limits<int>::max();
    }
    int delta_col = second.col - first.col;
    if (delta_col > mask_.width / 2) {
        delta_col -= mask_.width;
    } else if (delta_col < -mask_.width / 2) {
        delta_col += mask_.width;
    }
    const int delta_row = second.row - first.row;
    return delta_row * delta_row + delta_col * delta_col;
}

bool StitchConnectionOracle::empty_pixel(Pixel pixel) const
{
    if (!valid_pixel(pixel)) {
        return false;
    }
    const std::size_t index = static_cast<std::size_t>(pixel.row) * mask_.width
        + static_cast<std::size_t>(pixel.col);
    return component_ids_[index] < 0
        && !old_support_pixel(pixel)
        && !new_support_pixel(pixel);
}

int StitchConnectionOracle::compute_nearest_empty_distance_squared(Pixel start) const
{
    if (!valid_pixel(start)) {
        return std::numeric_limits<int>::max();
    }
    int best = std::numeric_limits<int>::max();
    const int radius = config_.stitch_connection_max_ri_radius;
    for (int dr = -radius; dr <= radius; ++dr) {
        const int row = start.row + dr;
        if (row < 0 || row >= mask_.height) {
            continue;
        }
        for (int dc = -radius; dc <= radius; ++dc) {
            const Pixel candidate{row, wrap_column(start.col + dc)};
            if (empty_pixel(candidate)) {
                best = std::min(best, pixel_distance_squared(start, candidate));
            }
        }
    }
    return best;
}

int StitchConnectionOracle::nearest_empty_distance_squared(int vertex, Pixel start) const
{
    if (!valid_pixel(start) || vertex < 0
        || static_cast<std::size_t>(vertex) >= nearest_empty_by_vertex_.size()) {
        return std::numeric_limits<int>::max();
    }
    int& cached = nearest_empty_by_vertex_[static_cast<std::size_t>(vertex)];
    if (cached < 0) {
        cached = compute_nearest_empty_distance_squared(start);
    }
    return cached;
}

bool StitchConnectionOracle::old_support_pixel(Pixel pixel) const
{
    if (!old_support_available_ || !valid_pixel(pixel)) {
        return false;
    }
    const std::size_t index = static_cast<std::size_t>(pixel.row) * mask_.width
        + static_cast<std::size_t>(pixel.col);
    return mask_.old_support[index] != 0;
}

bool StitchConnectionOracle::new_support_pixel(Pixel pixel) const
{
    if (!new_support_available_ || !valid_pixel(pixel)) {
        return false;
    }
    const std::size_t index = static_cast<std::size_t>(pixel.row) * mask_.width
        + static_cast<std::size_t>(pixel.col);
    return mask_.new_support[index] != 0;
}

bool StitchConnectionOracle::segment_supported_by_old(
    Pixel first,
    Pixel second
) const
{
    if (!valid_pixel(first) || !valid_pixel(second)) {
        return false;
    }

    int delta_col = second.col - first.col;
    if (delta_col > mask_.width / 2) {
        delta_col -= mask_.width;
    } else if (delta_col < -mask_.width / 2) {
        delta_col += mask_.width;
    }
    const int target_col = first.col + delta_col;
    const int step_row = (second.row > first.row) - (second.row < first.row);
    const int step_col = (target_col > first.col) - (target_col < first.col);
    const int abs_row = std::abs(second.row - first.row);
    const int abs_col = std::abs(delta_col);
    int row = first.row;
    int col = first.col;
    const MeshScalar next_col_boundary = first.col + (step_col > 0 ? 0.5F : -0.5F);
    const MeshScalar next_row_boundary = first.row + (step_row > 0 ? 0.5F : -0.5F);
    const MeshScalar delta_col_t = abs_col == 0
        ? std::numeric_limits<MeshScalar>::infinity()
        : 1.0F / static_cast<MeshScalar>(abs_col);
    const MeshScalar delta_row_t = abs_row == 0
        ? std::numeric_limits<MeshScalar>::infinity()
        : 1.0F / static_cast<MeshScalar>(abs_row);
    MeshScalar next_col_t = abs_col == 0
        ? std::numeric_limits<MeshScalar>::infinity()
        : (next_col_boundary - first.col) / static_cast<MeshScalar>(delta_col);
    MeshScalar next_row_t = abs_row == 0
        ? std::numeric_limits<MeshScalar>::infinity()
        : (next_row_boundary - first.row) / static_cast<MeshScalar>(second.row - first.row);

    Pixel last = first;
    int appended = 1;
    bool supported = true;
    const auto append = [&](int sample_row, int sample_col) {
        const Pixel sample{sample_row, wrap_column(sample_col)};
        if (last.row == sample.row && last.col == sample.col) {
            return;
        }
        if (appended > 1 && !old_support_pixel(last)) {
            supported = false;
        }
        last = sample;
        ++appended;
    };
    while (row != second.row || col != target_col) {
        if (!supported) {
            return false;
        }
        if (next_col_t < next_row_t) {
            col += step_col;
            append(row, col);
            next_col_t += delta_col_t;
        } else if (next_row_t < next_col_t) {
            row += step_row;
            append(row, col);
            next_row_t += delta_row_t;
        } else {
            // At an exact cell corner, include both side cells and the
            // diagonal cell so the support test cannot skip either side.
            append(row, col + step_col);
            append(row + step_row, col);
            col += step_col;
            row += step_row;
            append(row, col);
            next_col_t += delta_col_t;
            next_row_t += delta_row_t;
        }
    }
    return supported;
}

// Kept as one phase because run discovery, seam unions, and the final
// row-major remap share the same temporary run table; splitting them would
// require copying that table or exposing a private intermediate type.
void StitchConnectionOracle::build_components()
{
    const std::size_t pixel_count = static_cast<std::size_t>(mask_.height)
        * static_cast<std::size_t>(mask_.width);
    component_ids_.assign(pixel_count, -1);
    if (mask_.height <= 0 || mask_.width <= 0 || mask_.pixels.size() != pixel_count) {
        return;
    }

    struct RowRun {
        int start{};
        int end{};
        int label{-1};
    };
    std::vector<std::vector<RowRun>> row_runs(static_cast<std::size_t>(mask_.height));

    // A row scan is independent from every other row. Runs are sufficient to
    // describe 8-connectivity: adjacent rows only need interval-overlap
    // checks, avoiding one BFS queue and one union-find entry per RI pixel.
#pragma omp parallel for if (mask_.height > 32) schedule(static)
    for (int row = 0; row < mask_.height; ++row) {
        std::vector<RowRun>& runs = row_runs[static_cast<std::size_t>(row)];
        bool in_run = false;
        int run_start = 0;
        for (int col = 0; col <= mask_.width; ++col) {
            const bool active = col < mask_.width
                && mask_.pixels[static_cast<std::size_t>(row) * mask_.width
                    + static_cast<std::size_t>(col)] != 0;
            if (active && !in_run) {
                in_run = true;
                run_start = col;
            } else if (!active && in_run) {
                runs.push_back(RowRun{run_start, col - 1, -1});
                in_run = false;
            }
        }
    }

    int run_count = 0;
    for (std::vector<RowRun>& runs : row_runs) {
        for (RowRun& run : runs) {
            run.label = run_count++;
        }
    }
    std::vector<int> parent(static_cast<std::size_t>(run_count));
    std::iota(parent.begin(), parent.end(), 0);
    const auto find_root = [&parent](int value) {
        while (parent[static_cast<std::size_t>(value)] != value) {
            value = parent[static_cast<std::size_t>(value)];
        }
        return value;
    };
    const auto merge_labels = [&](int first, int second) {
        int first_root = find_root(first);
        int second_root = find_root(second);
        if (first_root == second_root) {
            return;
        }
        if (first_root > second_root) {
            std::swap(first_root, second_root);
        }
        parent[static_cast<std::size_t>(second_root)] = first_root;
    };

    const auto runs_touch = [&](const RowRun& first, const RowRun& second) {
        const bool ordinary_overlap = first.start <= second.end + 1
            && second.start <= first.end + 1;
        const bool wrapped_overlap =
            (first.start == 0 && second.end == mask_.width - 1)
            || (second.start == 0 && first.end == mask_.width - 1);
        return ordinary_overlap || wrapped_overlap;
    };
    for (int row = 0; row < mask_.height; ++row) {
        const std::vector<RowRun>& runs = row_runs[static_cast<std::size_t>(row)];
        if (runs.size() > 1 && runs.front().start == 0
            && runs.back().end == mask_.width - 1) {
            merge_labels(runs.front().label, runs.back().label);
        }
        if (row == 0) {
            continue;
        }
        const std::vector<RowRun>& previous = row_runs[static_cast<std::size_t>(row - 1)];
        std::size_t first_candidate = 0;
        for (const RowRun& current : runs) {
            while (first_candidate < previous.size()
                && previous[first_candidate].end + 1 < current.start) {
                ++first_candidate;
            }
            for (std::size_t index = first_candidate; index < previous.size(); ++index) {
                if (previous[index].start > current.end + 1) {
                    break;
                }
                if (runs_touch(current, previous[index])) {
                    merge_labels(current.label, previous[index].label);
                }
            }
        }
        // The linear two-pointer sweep cannot retain candidates separated by
        // the RI seam. Check the two possible 0 <-> width-1 connections
        // explicitly so diagonal 8-neighbour links survive horizontal wrap.
        if (!runs.empty() && !previous.empty()) {
            if (runs.front().start == 0 && previous.back().end == mask_.width - 1) {
                merge_labels(runs.front().label, previous.back().label);
            }
            if (runs.back().end == mask_.width - 1 && previous.front().start == 0) {
                merge_labels(runs.back().label, previous.front().label);
            }
        }
    }

    // Assign compact IDs by the first RI pixel of each component. This keeps
    // the historical row-major tie order. The remap pass itself is parallel.
    std::vector<int> root_to_component(static_cast<std::size_t>(run_count), -1);
    int next_component = 0;
    for (const std::vector<RowRun>& runs : row_runs) {
        for (const RowRun& run : runs) {
            const int root = find_root(run.label);
            if (root_to_component[static_cast<std::size_t>(root)] < 0) {
                root_to_component[static_cast<std::size_t>(root)] = next_component++;
            }
        }
    }

#pragma omp parallel for if (mask_.height > 32) schedule(static)
    for (int row = 0; row < mask_.height; ++row) {
        const std::vector<RowRun>& runs = row_runs[static_cast<std::size_t>(row)];
        for (const RowRun& run : runs) {
            const int component = root_to_component[
                static_cast<std::size_t>(find_root(run.label))
            ];
            for (int col = run.start; col <= run.end; ++col) {
                component_ids_[static_cast<std::size_t>(row) * mask_.width
                    + static_cast<std::size_t>(col)] = component;
            }
        }
    }
}

void StitchConnectionOracle::build_linear_support_offsets()
{
    const int radius = std::max(0, config_.stitch_connection_max_ri_radius);
    linear_support_offsets_.reserve(
        static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1))
    );
    for (int dr = -radius; dr <= radius; ++dr) {
        for (int dc = -radius; dc <= radius; ++dc) {
            linear_support_offsets_.push_back(Pixel{dr, dc});
        }
    }
    std::stable_sort(
        linear_support_offsets_.begin(),
        linear_support_offsets_.end(),
        [&](Pixel first, Pixel second) {
            const auto distance_squared = [&](Pixel offset) {
                int delta_col = offset.col;
                if (delta_col > mask_.width / 2) {
                    delta_col -= mask_.width;
                } else if (delta_col < -mask_.width / 2) {
                    delta_col += mask_.width;
                }
                return offset.row * offset.row + delta_col * delta_col;
            };
            return distance_squared(first) < distance_squared(second);
        }
    );
}

int StitchConnectionOracle::component_for_open_side(
    int vertex,
    MeshScalar angular_x,
    MeshScalar angular_y,
    bool source_is_old
) const
{
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertices_.size()) {
        return -1;
    }
    const Pixel start = projected_pixel_for_vertex(vertex);
    if (!valid_pixel(start)) {
        return -1;
    }
    const std::size_t start_index = static_cast<std::size_t>(start.row) * mask_.width
        + static_cast<std::size_t>(start.col);
    if (component_ids_[start_index] >= 0) {
        return component_ids_[start_index];
    }
    const Vec3f point = vertices_[static_cast<std::size_t>(vertex)];
    const MeshScalar elevation = std::atan2(
        point.z,
        std::hypot(point.x, point.y)
    );
    const MeshScalar horizontal_scale = std::max(static_cast<MeshScalar>(1e-6), std::cos(elevation));
    const MeshScalar fov_range = config_.vertical_fov.max_rad - config_.vertical_fov.min_rad;
    if (fov_range <= 0.0F || mask_.height <= 0 || mask_.width <= 0) {
        return -1;
    }
    const MeshScalar pixel_x = angular_x / horizontal_scale
        * static_cast<MeshScalar>(mask_.width) / (static_cast<MeshScalar>(2) * kPi);
    const MeshScalar pixel_y = angular_y / fov_range * static_cast<MeshScalar>(mask_.height);
    const MeshScalar pixel_length = std::hypot(pixel_x, pixel_y);
    if (pixel_length <= static_cast<MeshScalar>(1e-9)) {
        return -1;
    }
    const MeshScalar step_x = pixel_x / pixel_length;
    const MeshScalar step_y = pixel_y / pixel_length;
    Pixel previous = start;
    const int radius = std::max(0, config_.stitch_connection_max_ri_radius);
    for (int step = 1; step <= radius; ++step) {
        const Pixel sample{
            static_cast<int>(std::round(static_cast<MeshScalar>(start.row) + step_y * step)),
            wrap_column(static_cast<int>(std::round(
                static_cast<MeshScalar>(start.col) + step_x * step
            ))),
        };
        if (sample.row == previous.row && sample.col == previous.col) {
            continue;
        }
        previous = sample;
        if (!valid_pixel(sample)) {
            return -1;
        }
        const std::size_t index = static_cast<std::size_t>(sample.row) * mask_.width
            + static_cast<std::size_t>(sample.col);
        if (component_ids_[index] >= 0) {
            return component_ids_[index];
        }
        const bool old_support = old_support_pixel(sample);
        const bool new_support = new_support_pixel(sample);
        const bool own_support = source_is_old ? old_support : new_support;
        const bool other_support = source_is_old ? new_support : old_support;
        if (other_support || !own_support) {
            return -1;
        }
    }
    return -1;
}

void StitchConnectionOracle::collect_nearby_supports(
    Pixel pixel,
    std::vector<ComponentSupport>& supports
) const
{
    if (!valid_pixel(pixel)) {
        return;
    }
    for (int dr = -1; dr <= 1; ++dr) {
        const int row = pixel.row + dr;
        if (row < 0 || row >= mask_.height) {
            continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
            const Pixel candidate{row, wrap_column(pixel.col + dc)};
            const int component = component_ids_[
                static_cast<std::size_t>(candidate.row) * mask_.width
                + static_cast<std::size_t>(candidate.col)
            ];
            if (component < 0 || std::any_of(
                    supports.begin(), supports.end(), [&](const ComponentSupport& support) {
                        return support.component == component && support.anchor.row == candidate.row
                            && support.anchor.col == candidate.col;
                    })) {
                continue;
            }
            supports.push_back(ComponentSupport{component, candidate});
        }
    }
}

void StitchConnectionOracle::add_vertex_component(
    int vertex,
    int component,
    Pixel anchor
)
{
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertices_.size()
        || component < 0 || !valid_pixel(anchor)
        || component_ids_[static_cast<std::size_t>(anchor.row) * mask_.width
            + static_cast<std::size_t>(anchor.col)] != component) {
        return;
    }
    ComponentSupport& support = vertex_components_[static_cast<std::size_t>(vertex)];
    const bool has_support = support.component >= 0;
    const Pixel start = has_support
        ? support.vertex_pixel
        : projected_pixels_[static_cast<std::size_t>(vertex)];
    const int candidate_radius = pixel_chebyshev_distance(start, anchor);
    if (candidate_radius > config_.stitch_connection_max_ri_radius) {
        return;
    }
    int nearest_empty = has_support ? support.nearest_empty_distance_squared : -1;
    if (nearest_empty < 0) {
        nearest_empty = nearest_empty_distance_squared(vertex, start);
    }
    // Empty pixels are valid blockers. A red component wins only when it is
    // at least as close as the nearest unsupported/empty pixel.
    if (nearest_empty < pixel_distance_squared(start, anchor)) {
        return;
    }
    if (!has_support) {
        support = ComponentSupport{component, anchor, start, nearest_empty};
        return;
    }
    const int current_distance = pixel_distance_squared(start, support.anchor);
    if (pixel_distance_squared(start, anchor) < current_distance) {
        support.component = component;
        support.anchor = anchor;
    }
}

void StitchConnectionOracle::add_edge_support(
    int first,
    int second,
    std::vector<ComponentSupport>& edge_support_scratch
)
{
    if (first < 0 || second < 0
        || static_cast<std::size_t>(first) >= vertices_.size()
        || static_cast<std::size_t>(second) >= vertices_.size()) {
        return;
    }
    const Pixel start = projected_pixel_for_vertex(first);
    const Pixel end = projected_pixel_for_vertex(second);
    if (!valid_pixel(start) || !valid_pixel(end)) {
        return;
    }
    int delta_col = end.col - start.col;
    if (delta_col > mask_.width / 2) {
        delta_col -= mask_.width;
    } else if (delta_col < -mask_.width / 2) {
        delta_col += mask_.width;
    }
    const int steps = std::max(std::abs(end.row - start.row), std::abs(delta_col));
    edge_support_scratch.clear();
    std::vector<ComponentSupport>& supports = edge_support_scratch;
    Pixel previous_sample{-1, -1};
    for (int step = 0; step <= steps; ++step) {
        const MeshScalar fraction = steps == 0 ? 0.0F : static_cast<MeshScalar>(step) / steps;
        const Pixel sample{
            static_cast<int>(std::llround(start.row + fraction * (end.row - start.row))),
            wrap_column(static_cast<int>(std::llround(start.col + fraction * delta_col))),
        };
        if (sample.row == previous_sample.row && sample.col == previous_sample.col) {
            continue;
        }
        previous_sample = sample;
        collect_nearby_supports(sample, supports);
    }
    for (const ComponentSupport& support : supports) {
        add_vertex_component(first, support.component, support.anchor);
        add_vertex_component(second, support.component, support.anchor);
    }
}

void StitchConnectionOracle::add_boundary_support(
    const std::vector<BoundaryPath>& paths
)
{
    std::vector<ComponentSupport> edge_support_scratch;
    edge_support_scratch.reserve(32);
    for (const BoundaryPath& path : paths) {
        for (const int vertex : path.ids) {
            add_linear_frontier_support(vertex);
        }
        for (std::size_t idx = 1; idx < path.ids.size(); ++idx) {
            add_edge_support(path.ids[idx - 1], path.ids[idx], edge_support_scratch);
        }
        if (path.closed && path.ids.size() > 2) {
            add_edge_support(path.ids.back(), path.ids.front(), edge_support_scratch);
        }
    }
}

void StitchConnectionOracle::prepare_nearest_empty_cache(
    const std::vector<BoundaryPath>& old_paths,
    const std::vector<BoundaryPath>& new_paths
)
{
    if (nearest_empty_by_vertex_.empty()) {
        return;
    }
    std::vector<int> start_vertices;
    const auto collect_vertices = [&](const std::vector<BoundaryPath>& paths) {
        for (const BoundaryPath& path : paths) {
            for (const int vertex : path.ids) {
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= projected_pixels_.size()) {
                    continue;
                }
                const Pixel start = projected_pixels_[static_cast<std::size_t>(vertex)];
                if (valid_pixel(start)) {
                    start_vertices.push_back(vertex);
                }
            }
        }
    };
    collect_vertices(old_paths);
    collect_vertices(new_paths);
    std::sort(start_vertices.begin(), start_vertices.end());
    start_vertices.erase(
        std::unique(start_vertices.begin(), start_vertices.end()),
        start_vertices.end()
    );
#if defined(RI_MESHING_HAS_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int index = 0; index < static_cast<int>(start_vertices.size()); ++index) {
        const int vertex = start_vertices[static_cast<std::size_t>(index)];
        int& cached = nearest_empty_by_vertex_[static_cast<std::size_t>(vertex)];
        if (cached < 0) {
            cached = compute_nearest_empty_distance_squared(
                projected_pixels_[static_cast<std::size_t>(vertex)]
            );
        }
    }
}

void StitchConnectionOracle::add_linear_frontier_support(int vertex)
{
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertices_.size()
        || config_.stitch_connection_max_ri_radius <= 0
        || !mask_pixels_available_ || !old_support_available_) {
        return;
    }
    const Pixel start = projected_pixel_for_vertex(vertex);
    if (!valid_pixel(start)) {
        return;
    }

    for (const Pixel offset : linear_support_offsets_) {
        const int row = start.row + offset.row;
        if (row < 0 || row >= mask_.height) {
            continue;
        }
        const Pixel candidate{row, wrap_column(start.col + offset.col)};
        const std::size_t index = static_cast<std::size_t>(candidate.row) * mask_.width
            + static_cast<std::size_t>(candidate.col);
        const int candidate_component = component_ids_[index];
        if (candidate_component < 0 || !mask_.pixels[index]
            || !segment_supported_by_old(start, candidate)) {
            continue;
        }
        add_vertex_component(vertex, candidate_component, candidate);
        return;
    }
}

bool StitchConnectionOracle::segment_touches_component(
    Pixel start,
    Pixel end,
    int component
) const
{
    if (!valid_pixel(start) || !valid_pixel(end) || component < 0) {
        return false;
    }
    int delta_col = end.col - start.col;
    if (delta_col > mask_.width / 2) {
        delta_col -= mask_.width;
    } else if (delta_col < -mask_.width / 2) {
        delta_col += mask_.width;
    }
    const int steps = std::max(std::abs(end.row - start.row), std::abs(delta_col));
    for (int step = 0; step <= steps; ++step) {
        const MeshScalar fraction = steps == 0 ? 0.0F : static_cast<MeshScalar>(step) / steps;
        const Pixel sample{
            static_cast<int>(std::llround(start.row + fraction * (end.row - start.row))),
            wrap_column(static_cast<int>(std::llround(start.col + fraction * delta_col))),
        };
        for (int dr = -1; dr <= 1; ++dr) {
            const int row = sample.row + dr;
            if (row < 0 || row >= mask_.height) {
                continue;
            }
            for (int dc = -1; dc <= 1; ++dc) {
                const int col = wrap_column(sample.col + dc);
                if (component_ids_[static_cast<std::size_t>(row) * mask_.width
                        + static_cast<std::size_t>(col)] == component) {
                    return true;
                }
            }
        }
    }
    return false;
}

StitchConnectionOracle::Pixel StitchConnectionOracle::nearest_component_pixel(
    Pixel start,
    int component
) const
{
    Pixel best;
    if (!valid_pixel(start) || component < 0) {
        return best;
    }
    int best_squared_distance = std::numeric_limits<int>::max();
    const int radius = config_.stitch_connection_max_ri_radius;
    for (int dr = -radius; dr <= radius; ++dr) {
        const int row = start.row + dr;
        if (row < 0 || row >= mask_.height) {
            continue;
        }
        for (int dc = -radius; dc <= radius; ++dc) {
            const Pixel candidate{row, wrap_column(start.col + dc)};
            const int candidate_component = component_ids_[
                static_cast<std::size_t>(candidate.row) * mask_.width
                + static_cast<std::size_t>(candidate.col)
            ];
            if (candidate_component != component) {
                continue;
            }
            const int squared_distance = pixel_distance_squared(start, candidate);
            if (squared_distance < best_squared_distance) {
                best = candidate;
                best_squared_distance = squared_distance;
            }
        }
    }
    return best;
}

bool StitchConnectionOracle::within_component_red_steps_fast(
    Pixel first,
    Pixel second,
    int component
) const
{
    constexpr std::size_t kFastCapacity = 64;
    std::array<int, kFastCapacity> frontier{};
    std::array<int, kFastCapacity> next{};
    std::array<int, kFastCapacity> visited{};
    int frontier_count = 1;
    int visited_count = 1;
    frontier[0] = first.row * mask_.width + first.col;
    visited[0] = frontier[0];
    const int target = second.row * mask_.width + second.col;
    for (int step = 0; step < config_.stitch_connection_max_red_steps; ++step) {
        int next_count = 0;
        for (int index = 0; index < frontier_count; ++index) {
            const int current = frontier[static_cast<std::size_t>(index)];
            const int current_row = current / mask_.width;
            const int current_col = current % mask_.width;
            for (int dr = -1; dr <= 1; ++dr) {
                const int row = current_row + dr;
                if (row < 0 || row >= mask_.height) {
                    continue;
                }
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    const int col = wrap_column(current_col + dc);
                    const int flat = row * mask_.width + col;
                    if (component_ids_[static_cast<std::size_t>(flat)] != component) {
                        continue;
                    }
                    bool already_visited = false;
                    for (int visited_index = 0; visited_index < visited_count; ++visited_index) {
                        if (visited[static_cast<std::size_t>(visited_index)] == flat) {
                            already_visited = true;
                            break;
                        }
                    }
                    if (already_visited) {
                        continue;
                    }
                    visited[static_cast<std::size_t>(visited_count++)] = flat;
                    if (flat == target) {
                        return true;
                    }
                    next[static_cast<std::size_t>(next_count++)] = flat;
                }
            }
        }
        frontier = next;
        frontier_count = next_count;
        if (frontier_count == 0) {
            return false;
        }
    }
    return false;
}

bool StitchConnectionOracle::within_component_red_steps(
    Pixel first,
    Pixel second,
    int component
) const
{
    if (!valid_pixel(first) || !valid_pixel(second) || component < 0
        || component_ids_[static_cast<std::size_t>(first.row) * mask_.width
            + static_cast<std::size_t>(first.col)] != component
        || component_ids_[static_cast<std::size_t>(second.row) * mask_.width
            + static_cast<std::size_t>(second.col)] != component) {
        return false;
    }
    if (first.row == second.row && first.col == second.col) {
        return true;
    }
    const int max_steps = config_.stitch_connection_max_red_steps;
    if (max_steps <= 3) {
        return within_component_red_steps_fast(first, second, component);
    }
    std::vector<Pixel> frontier{first};
    std::vector<Pixel> next;
    std::unordered_set<int> visited{
        first.row * mask_.width + first.col
    };
    for (int step = 0; step < max_steps; ++step) {
        next.clear();
        for (const Pixel current : frontier) {
            for (int dr = -1; dr <= 1; ++dr) {
                const int row = current.row + dr;
                if (row < 0 || row >= mask_.height) {
                    continue;
                }
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    const Pixel candidate{row, wrap_column(current.col + dc)};
                    const int flat = candidate.row * mask_.width + candidate.col;
                    if (component_ids_[static_cast<std::size_t>(flat)] != component
                        || !visited.insert(flat).second) {
                        continue;
                    }
                    if (candidate.row == second.row && candidate.col == second.col) {
                        return true;
                    }
                    next.push_back(candidate);
                }
            }
        }
        frontier.swap(next);
        if (frontier.empty()) {
            return false;
        }
    }
    return false;
}

int StitchConnectionOracle::component_for_pair_impl(
    int old_vertex,
    int new_vertex,
    bool enforce_red_steps
) const
{
    if (old_vertex < 0 || new_vertex < 0
        || static_cast<std::size_t>(old_vertex) >= vertices_.size()
        || static_cast<std::size_t>(new_vertex) >= vertices_.size()) {
        return -1;
    }
    const ComponentSupport& old_support =
        vertex_components_[static_cast<std::size_t>(old_vertex)];
    const ComponentSupport& new_support =
        vertex_components_[static_cast<std::size_t>(new_vertex)];
    if (old_support.component < 0
        || new_support.component < 0
        || old_support.component != new_support.component) {
        return -1;
    }
    const Pixel old_pixel = old_support.vertex_pixel;
    const Pixel new_pixel = new_support.vertex_pixel;
    if (!segment_touches_component(old_pixel, new_pixel, old_support.component)) {
        return -1;
    }
    if (enforce_red_steps && !within_component_red_steps(
            old_support.anchor, new_support.anchor, old_support.component)) {
        return -1;
    }
    return old_support.component;
}

int StitchConnectionOracle::component_for_pair(int old_vertex, int new_vertex) const
{
    return component_for_pair_impl(old_vertex, new_vertex, true);
}

bool StitchConnectionOracle::pair_allowed(int old_vertex, int new_vertex) const
{
    return component_for_pair(old_vertex, new_vertex) >= 0;
}

bool StitchConnectionOracle::pair_allowed_for_geometry(int old_vertex, int new_vertex) const
{
    return component_for_pair_impl(old_vertex, new_vertex, false) >= 0;
}

bool StitchConnectionOracle::face_allowed(Triangle face, int old_vertex_count) const
{
    const int ids[3]{face.v0, face.v1, face.v2};
    for (int idx = 0; idx < 3; ++idx) {
        const int first = ids[idx];
        const int second = ids[(idx + 1) % 3];
        const bool first_old = first < old_vertex_count;
        const bool second_old = second < old_vertex_count;
        if (first_old == second_old) {
            continue;
        }
        const int old_vertex = first_old ? first : second;
        const int new_vertex = first_old ? second : first;
        if (!pair_allowed_for_geometry(old_vertex, new_vertex)) {
            return false;
        }
    }
    return true;
}

}  // namespace ri_meshing::stitch_detail
