#include "ri_meshing/io.hpp"

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ri_meshing {

std::filesystem::path expand_user_path(const std::string& path)
{
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home == nullptr) {
            throw std::runtime_error("Cannot expand '~': HOME is not set");
        }
        if (path.size() == 1) {
            return std::filesystem::path(home);
        }
        if (path[1] == '/') {
            return std::filesystem::path(home) / path.substr(2);
        }
    }
    return std::filesystem::path(path);
}

std::filesystem::path frame_bin_path(
    const std::string& dataset_path,
    const std::string& sequence,
    int frame_index
)
{
    char filename[16]{};
    std::snprintf(filename, sizeof(filename), "%06d.bin", frame_index);
    return expand_user_path(dataset_path) / "sequences" / sequence / "velodyne" / filename;
}

namespace {

std::filesystem::path sequence_path(const std::string& dataset_path, const std::string& sequence)
{
    return expand_user_path(dataset_path) / "sequences" / sequence;
}

Mat4f identity_matrix()
{
    return Mat4f{1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0};
}

Mat4f parse_pose_values(const std::string& line)
{
    std::istringstream input(line);
    std::array<MeshScalar, 12> values{};
    for (MeshScalar& value : values) {
        if (!(input >> value)) {
            throw std::runtime_error("Malformed pose row: " + line);
        }
    }
    return Mat4f{values[0], values[1], values[2], values[3],
                 values[4], values[5], values[6], values[7],
                 values[8], values[9], values[10], values[11],
                 0.0, 0.0, 0.0, 1.0};
}

Mat4f multiply(Mat4f a, Mat4f b)
{
    Mat4f out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            MeshScalar value = static_cast<MeshScalar>(0);
            for (int k = 0; k < 4; ++k) {
                value += a[row * 4 + k] * b[k * 4 + col];
            }
            out[row * 4 + col] = value;
        }
    }
    return out;
}

Mat4f invert_rigid(Mat4f pose)
{
    Mat4f out = identity_matrix();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out[row * 4 + col] = pose[col * 4 + row];
        }
    }
    for (int row = 0; row < 3; ++row) {
        out[row * 4 + 3] = -(
            out[row * 4 + 0] * pose[3]
            + out[row * 4 + 1] * pose[7]
            + out[row * 4 + 2] * pose[11]
        );
    }
    return out;
}

Mat4f read_calib_transform(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open calibration file: " + path.string());
    }
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("Tr:", 0) == 0) {
            line.erase(0, 3);
            return parse_pose_values(line);
        }
    }
    return identity_matrix();
}

}  // namespace

std::vector<Vec3f> load_kitti_bin_points(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open point cloud: " + path.string());
    }

    std::vector<Vec3f> points;
    std::array<float, 4> record{};
    while (input.read(reinterpret_cast<char*>(record.data()), sizeof(record))) {
        const Vec3f point{record[0], record[1], record[2]};
        const MeshScalar range_sq = point.x * point.x + point.y * point.y + point.z * point.z;
        if (range_sq > static_cast<MeshScalar>(0)) {
            points.push_back(point);
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("Malformed point cloud: " + path.string());
    }
    return points;
}

Mat4f read_pose(
    const std::string& dataset_path,
    const std::string& sequence,
    int frame_index,
    bool use_custom_pose
)
{
    const std::filesystem::path seq = sequence_path(dataset_path, sequence);
    const std::filesystem::path pose_path = seq / (use_custom_pose ? "poses_custom.txt" : "poses.txt");
    std::ifstream input(pose_path);
    if (!input) {
        throw std::runtime_error("Cannot open pose file: " + pose_path.string());
    }

    std::string line;
    for (int idx = 0; idx <= frame_index; ++idx) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("Pose index out of range in: " + pose_path.string());
        }
    }
    Mat4f pose = parse_pose_values(line);
    if (use_custom_pose) {
        return pose;
    }

    const Mat4f cam_to_lidar = read_calib_transform(seq / "calib.txt");
    return multiply(multiply(invert_rigid(cam_to_lidar), pose), cam_to_lidar);
}

void write_ascii_ply(
    const std::filesystem::path& path,
    const std::vector<Vec3f>& vertices,
    const std::vector<Triangle>& faces
)
{
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create PLY: " + path.string());
    }

    output << "ply\nformat ascii 1.0\n";
    output << "element vertex " << vertices.size() << '\n';
    output << "property float x\nproperty float y\nproperty float z\n";
    output << "element face " << faces.size() << '\n';
    output << "property list uchar int vertex_indices\nend_header\n";
    output << std::setprecision(9);
    for (const Vec3f vertex : vertices) {
        output << vertex.x << ' ' << vertex.y << ' ' << vertex.z << '\n';
    }
    for (const Triangle face : faces) {
        output << "3 " << face.v0 << ' ' << face.v1 << ' ' << face.v2 << '\n';
    }
}

}  // namespace ri_meshing
