#include "ri_meshing/boundary_stitching.hpp"

#include "ri_meshing/boundary_paths.hpp"
#include "ri_meshing/timing.hpp"
#include "stitch_core_components.hpp"
#include "stitch_connection.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <utility>

namespace ri_meshing {
namespace {

using stitch_detail::Edge;

StitchDebugPath make_debug_path(const BoundaryPath& path)
{
    return StitchDebugPath{path.ids, path.closed};
}

void add_timing(PipelineTiming* timing, const char* name, const Stopwatch& stopwatch)
{
    if (timing != nullptr) {
        timing->add_stage(name, stopwatch.elapsed_milliseconds());
    }
}

template <typename Input>
class StitchCorePipeline {
public:
    StitchCorePipeline(const Input& input, const SparseMeshingConfig& config)
        : input_(input), config_(config)
    {
    }

    StitchCoreResult run()
    {
        if (!has_stitch_input()) {
            return result_;
        }
        prepare_boundaries();
        extract_paths();
        if (old_paths_.empty() || new_paths_.empty()) {
            return result_;
        }
        match_paths();
        generate_candidates();
        select_faces();
        return result_;
    }

private:
    bool has_stitch_input() const
    {
        return input_.old_vertex_count > 0
            && !input_.old_boundary_edges.empty()
            && !input_.new_boundary_edges.empty();
    }

    void prepare_boundaries()
    {
        Stopwatch stopwatch;
        old_boundary_edges_ = &input_.old_boundary_edges;
        result_.old_boundary_edges_used = old_boundary_edges_->size();

        new_boundary_edges_ = &input_.new_boundary_edges;
        if (input_.debug != nullptr) {
            input_.debug->old_boundary_edges = *old_boundary_edges_;
            input_.debug->new_boundary_edges = *new_boundary_edges_;
        }
        add_timing(input_.timing, "stitch.core.boundary_edges", stopwatch);
    }

    void extract_paths()
    {
        Stopwatch stopwatch;
        Stopwatch sub_stopwatch;
        if (input_.path_workspace != nullptr) {
            extract_boundary_paths(*old_boundary_edges_, old_paths_, input_.path_workspace);
        } else {
            old_paths_ = extract_boundary_paths(*old_boundary_edges_);
        }
        add_timing(input_.timing, "stitch.core.paths.old", sub_stopwatch);

        sub_stopwatch = Stopwatch{};
        if (input_.path_workspace != nullptr) {
            extract_boundary_paths(*new_boundary_edges_, new_paths_, input_.path_workspace);
        } else {
            new_paths_ = extract_boundary_paths(*new_boundary_edges_);
        }
        add_timing(input_.timing, "stitch.core.paths.new", sub_stopwatch);
        publish_path_debug();
        add_timing(input_.timing, "stitch.core.paths", stopwatch);

        result_.old_boundary_vertices = count_path_vertices(old_paths_);
        result_.new_boundary_vertices = count_path_vertices(new_paths_);
    }

    static std::size_t count_path_vertices(const std::vector<BoundaryPath>& paths)
    {
        return std::accumulate(
            paths.begin(),
            paths.end(),
            std::size_t{0},
            [](std::size_t sum, const BoundaryPath& path) { return sum + path.ids.size(); }
        );
    }

    void publish_path_debug()
    {
        if (input_.debug == nullptr) {
            return;
        }
        input_.debug->old_paths.clear();
        input_.debug->new_paths.clear();
        input_.debug->new_labels.clear();
        input_.debug->old_paths.reserve(old_paths_.size());
        input_.debug->new_paths.reserve(new_paths_.size());
        for (const BoundaryPath& path : old_paths_) {
            input_.debug->old_paths.push_back(make_debug_path(path));
        }
        publish_new_path_debug();
    }

    void publish_new_path_debug()
    {
        int next_global_label = 0;
        for (std::size_t path_idx = 0; path_idx < new_paths_.size(); ++path_idx) {
            const BoundaryPath& path = new_paths_[path_idx];
            input_.debug->new_paths.push_back(make_debug_path(path));
            const MeshScalar denom = path.ids.size() > 1
                ? static_cast<MeshScalar>(path.ids.size() - 1) : static_cast<MeshScalar>(1);
            for (std::size_t idx = 0; idx < path.ids.size(); ++idx) {
                input_.debug->new_labels.push_back(StitchDebugLabel{
                    path.ids[idx],
                    static_cast<int>(path_idx),
                    next_global_label,
                    static_cast<MeshScalar>(idx) / denom,
                    static_cast<MeshScalar>(0),
                });
                ++next_global_label;
            }
        }
    }

    void match_paths()
    {
        if (input_.connection_mask != nullptr && !input_.connection_mask->empty()) {
            connection_oracle_.emplace(
                stitch_detail::StitchConnectionOracleRequest{
                    *input_.connection_mask,
                    config_,
                    input_.vertices_local,
                    old_paths_,
                    new_paths_,
                    input_.timing,
                }
            );
        }
        matching_ = stitch_detail::match_boundaries(
            stitch_detail::BoundaryMatchingRequest{
                input_.vertices_local,
                *old_boundary_edges_,
                *new_boundary_edges_,
                input_.vertex_normals_local,
                old_paths_,
                new_paths_,
                input_.old_vertex_count,
                true,
                connection_oracle_ ? &*connection_oracle_ : nullptr,
            },
            config_,
            input_.debug,
            input_.timing
        );
    }

    void generate_candidates()
    {
        candidates_ = stitch_detail::generate_stitch_candidates(
            stitch_detail::CandidateGenerationRequest{
                new_paths_,
                matching_.old_segments_by_cluster,
                input_.vertices_local,
                matching_.old_incidence,
                matching_.new_incidence,
                config_,
                input_.stitch_proxy_edges_callback,
                connection_oracle_ ? &*connection_oracle_ : nullptr,
                &old_paths_,
                input_.ownership_workspace,
            },
            input_.debug,
            input_.timing
        );
    }

    void select_faces()
    {
        stitch_detail::ManifoldSelectionResult selection = stitch_detail::select_stitch_faces(
            stitch_detail::ManifoldSelectionRequest{
                std::move(candidates_),
                input_.vertices_local,
                *old_boundary_edges_,
                *new_boundary_edges_,
                input_.old_vertex_count,
                connection_oracle_ ? &*connection_oracle_ : nullptr,
            },
            config_,
            input_.debug,
            input_.timing
        );
        result_.candidate_faces = selection.candidate_faces;
        result_.faces = std::move(selection.faces);
    }

    const Input& input_;
    const SparseMeshingConfig& config_;
    StitchCoreResult result_;
    const std::vector<Edge>* old_boundary_edges_{};
    const std::vector<Edge>* new_boundary_edges_{};
    std::vector<BoundaryPath> old_paths_;
    std::vector<BoundaryPath> new_paths_;
    stitch_detail::BoundaryMatchingResult matching_;
    std::vector<Triangle> candidates_;
    std::optional<stitch_detail::StitchConnectionOracle> connection_oracle_;
};

}  // namespace

StitchCoreResult stitch_core(const StitchCoreInput& input, const SparseMeshingConfig& config)
{
    return StitchCorePipeline(input, config).run();
}

}  // namespace ri_meshing
