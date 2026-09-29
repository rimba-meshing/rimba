#include "ri_meshing/io.hpp"

#include <cassert>
#include <filesystem>

int main()
{
    const auto path = ri_meshing::frame_bin_path("~/Datasets/local/mai_city/bin", "00", 0);
    if (!std::filesystem::exists(path)) {
        return 0;
    }

    const auto points = ri_meshing::load_kitti_bin_points(path);
    assert(points.size() == 71044);
    return 0;
}
