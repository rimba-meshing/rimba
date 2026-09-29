#pragma once

#include <cmath>
#include <type_traits>

namespace ri_meshing::cuda_detail::accumulation_internal {

struct DeviceMat4 {
    MeshScalar m[16];
};

static_assert(std::is_trivially_copyable_v<DeviceMat4>);

inline DeviceMat4 to_device_mat(Mat4f mat)
{
    DeviceMat4 out{};
    for (int idx = 0; idx < 16; ++idx) {
        out.m[idx] = mat[idx];
    }
    return out;
}

__device__ inline Vec3f transform_point(Vec3f point, DeviceMat4 transform)
{
    const MeshScalar x = transform.m[0] * point.x + transform.m[1] * point.y + transform.m[2] * point.z + transform.m[3];
    const MeshScalar y = transform.m[4] * point.x + transform.m[5] * point.y + transform.m[6] * point.z + transform.m[7];
    const MeshScalar z = transform.m[8] * point.x + transform.m[9] * point.y + transform.m[10] * point.z + transform.m[11];
    const MeshScalar w = transform.m[12] * point.x + transform.m[13] * point.y + transform.m[14] * point.z + transform.m[15];
    const MeshScalar inv_w = fabs(w) > static_cast<MeshScalar>(1e-12)
        ? static_cast<MeshScalar>(1) / w : static_cast<MeshScalar>(1);
    return Vec3f{static_cast<float>(x * inv_w), static_cast<float>(y * inv_w), static_cast<float>(z * inv_w)};
}

}  // namespace ri_meshing::cuda_detail::accumulation_internal
