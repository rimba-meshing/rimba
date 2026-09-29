#pragma once

#include "ri_meshing/types.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace ri_meshing {

#if defined(__CUDACC__)
#define RI_MESHING_FACE_ORIENTATION_HOST_DEVICE __host__ __device__
#else
#define RI_MESHING_FACE_ORIENTATION_HOST_DEVICE
#endif

// Negative values mean that the face normal points towards the sensor origin.
// The helper is host/device so dense CUDA emission and host stitching share the
// same orientation contract without duplicating the cross-product arithmetic.
RI_MESHING_FACE_ORIENTATION_HOST_DEVICE inline MeshScalar sensor_orientation_dot(
    Vec3f a,
    Vec3f b,
    Vec3f c
) noexcept
{
    const MeshScalar ux = static_cast<MeshScalar>(b.x) - a.x;
    const MeshScalar uy = static_cast<MeshScalar>(b.y) - a.y;
    const MeshScalar uz = static_cast<MeshScalar>(b.z) - a.z;
    const MeshScalar vx = static_cast<MeshScalar>(c.x) - a.x;
    const MeshScalar vy = static_cast<MeshScalar>(c.y) - a.y;
    const MeshScalar vz = static_cast<MeshScalar>(c.z) - a.z;
    const MeshScalar nx = uy * vz - uz * vy;
    const MeshScalar ny = uz * vx - ux * vz;
    const MeshScalar nz = ux * vy - uy * vx;
    const MeshScalar cx = (static_cast<MeshScalar>(a.x) + b.x + c.x) / static_cast<MeshScalar>(3);
    const MeshScalar cy = (static_cast<MeshScalar>(a.y) + b.y + c.y) / static_cast<MeshScalar>(3);
    const MeshScalar cz = (static_cast<MeshScalar>(a.z) + b.z + c.z) / static_cast<MeshScalar>(3);
    return nx * cx + ny * cy + nz * cz;
}

inline bool orient_face_toward_sensor(
    Triangle& face,
    const std::vector<Vec3f>& vertices
) noexcept
{
    if (face.v0 < 0 || face.v1 < 0 || face.v2 < 0
        || static_cast<std::size_t>(face.v0) >= vertices.size()
        || static_cast<std::size_t>(face.v1) >= vertices.size()
        || static_cast<std::size_t>(face.v2) >= vertices.size()) {
        return false;
    }
    if (sensor_orientation_dot(
            vertices[static_cast<std::size_t>(face.v0)],
            vertices[static_cast<std::size_t>(face.v1)],
            vertices[static_cast<std::size_t>(face.v2)]) > 0.0) {
        std::swap(face.v1, face.v2);
        return true;
    }
    return false;
}

#undef RI_MESHING_FACE_ORIENTATION_HOST_DEVICE

}  // namespace ri_meshing
