#pragma once

#include <thrust/device_vector.h>

#include <algorithm>
#include <cstddef>

namespace ri_meshing::cuda_detail {

// Keep capacity growth below the default doubling behavior. This changes only
// allocation timing and spare capacity, not vector contents or iteration order.
template <typename T>
void resize_with_headroom(thrust::device_vector<T>& values, std::size_t requested)
{
    if (requested > values.capacity()) {
        const std::size_t current = values.capacity();
        const std::size_t grown = current == 0
            ? requested
            : current + std::max<std::size_t>(current / 2, 1);
        values.reserve(std::max(requested, grown));
    }
    values.resize(requested);
}

}  // namespace ri_meshing::cuda_detail
