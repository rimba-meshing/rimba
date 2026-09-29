#pragma once

#include "ri_meshing/types.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace ri_meshing {

std::filesystem::path expand_user_path(const std::string& path);
std::filesystem::path frame_bin_path(
    const std::string& dataset_path,
    const std::string& sequence,
    int frame_index
);

std::vector<Vec3f> load_kitti_bin_points(const std::filesystem::path& path);
Mat4f read_pose(
    const std::string& dataset_path,
    const std::string& sequence,
    int frame_index,
    bool use_custom_pose = false
);
void write_ascii_ply(
    const std::filesystem::path& path,
    const std::vector<Vec3f>& vertices,
    const std::vector<Triangle>& faces
);

}  // namespace ri_meshing
