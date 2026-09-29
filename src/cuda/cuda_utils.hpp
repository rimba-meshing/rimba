#pragma once

#include "ri_meshing/timing.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstddef>
#include <string>
#include <stdexcept>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t error = (call);                                                          \
        if (error != cudaSuccess) {                                                                \
            throw std::runtime_error(cudaGetErrorString(error));                                   \
        }                                                                                          \
    } while (false)

namespace ri_meshing::cuda_detail {

template <typename T>
inline void copy_device_to_device_async(
    T* destination,
    const T* source,
    std::size_t count,
    cudaStream_t stream = 0
)
{
    if (count == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        destination,
        source,
        count * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream
    ));
}

template <typename T>
inline void copy_host_to_device_sync(
    T* destination,
    const T* source,
    std::size_t count,
    cudaStream_t stream = 0
)
{
    if (count == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        destination,
        source,
        count * sizeof(T),
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
inline void copy_device_to_host_sync(
    T* destination,
    const T* source,
    std::size_t count,
    cudaStream_t stream = 0
)
{
    if (count == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        destination,
        source,
        count * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

inline int scanned_count(thrust::device_vector<int>& flags, thrust::device_vector<int>& offsets, cudaStream_t stream = 0)
{
    if (flags.empty()) {
        return 0;
    }
    thrust::exclusive_scan(thrust::cuda::par.on(stream), flags.begin(), flags.end(), offsets.begin());
    int last_flag = 0;
    int last_offset = 0;
    const int last = static_cast<int>(flags.size()) - 1;
    CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        thrust::raw_pointer_cast(flags.data()) + last,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        thrust::raw_pointer_cast(offsets.data()) + last,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return last_flag + last_offset;
}

template <typename Fn>
auto timed_cuda_stage(PipelineTiming* timing, const std::string& name, Fn&& fn)
{
#if !defined(RI_MESHING_ENABLE_DETAILED_TIMING)
    (void)timing;
    (void)name;
    auto result = fn();
    CUDA_CHECK(cudaGetLastError());
    return result;
#else
    if (timing == nullptr) {
        auto result = fn();
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (timing->synchronize_cuda) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const Stopwatch stopwatch;
    auto result = fn();
    if (timing->synchronize_cuda) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timing->add_stage(name, stopwatch.elapsed_milliseconds());
    return result;
#endif
}

template <typename Fn>
void timed_cuda_stage_void(PipelineTiming* timing, const std::string& name, Fn&& fn)
{
#if !defined(RI_MESHING_ENABLE_DETAILED_TIMING)
    (void)timing;
    (void)name;
    fn();
    CUDA_CHECK(cudaGetLastError());
#else
    if (timing == nullptr) {
        fn();
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (timing->synchronize_cuda) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const Stopwatch stopwatch;
    fn();
    if (timing->synchronize_cuda) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timing->add_stage(name, stopwatch.elapsed_milliseconds());
#endif
}

}  // namespace ri_meshing::cuda_detail
