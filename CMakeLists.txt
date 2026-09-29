cmake_minimum_required(VERSION 4.2)
include(CheckLanguage)

project(
    ri_meshing
    VERSION 0.1.0
    LANGUAGES CXX
)

option(RI_MESHING_ENABLE_OPENMP "Enable OpenMP CPU parallelism when available" ON)
option(RI_MESHING_ENABLE_CUDA "Build CUDA kernels and GPU backend" OFF)
option(RI_MESHING_ENABLE_DETAILED_TIMING "Enable detailed per-stage timing instrumentation" ON)
option(RI_MESHING_ENABLE_DEBUG_EXPORT "Enable heavyweight debug snapshot export paths" ON)
option(RI_MESHING_ENABLE_MEMORY_INSTRUMENTATION "Enable per-stage host/device memory accounting" OFF)
option(RI_MESHING_BUILD_TESTS "Build smoke and unit tests" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(RI_MESHING_ENABLE_CUDA)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "RI_MESHING_ENABLE_CUDA=ON requires the CUDA toolkit and nvcc.")
    endif()
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(ri_meshing_core)
target_sources(
    ri_meshing_core
    PRIVATE
        src/boundary_graph.cpp
        src/boundary_paths.cpp
        src/boundary_stitching.cpp
        src/config.cpp
        src/geometry.cpp
        src/hole_filling.cpp
        src/io.cpp
        src/stitch_candidates.cpp
        src/stitch_boundary_filters.cpp
        src/stitch_connection.cpp
        src/stitch_manifold.cpp
        src/stitch_matching.cpp
)
target_include_directories(
    ri_meshing_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(ri_meshing_core PUBLIC cxx_std_23)

if(RI_MESHING_ENABLE_DETAILED_TIMING)
    target_compile_definitions(ri_meshing_core PUBLIC RI_MESHING_ENABLE_DETAILED_TIMING=1)
endif()

if(RI_MESHING_ENABLE_DEBUG_EXPORT)
    target_compile_definitions(ri_meshing_core PUBLIC RI_MESHING_ENABLE_DEBUG_EXPORT=1)
endif()

if(RI_MESHING_ENABLE_MEMORY_INSTRUMENTATION)
    target_compile_definitions(ri_meshing_core PUBLIC RI_MESHING_ENABLE_MEMORY_INSTRUMENTATION=1)
endif()

if(RI_MESHING_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ri_meshing_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(ri_meshing_core PUBLIC RI_MESHING_HAS_OPENMP=1)
    endif()
endif()

if(RI_MESHING_ENABLE_CUDA)
    target_sources(
        ri_meshing_core
        PRIVATE
            src/cuda/accumulation.cu
            src/cuda/accumulation_append.cu
            src/cuda/frame_vertex_compaction.cu
            src/cuda/coarsened_dense_meshing.cu
            src/cuda/dense_candidates.cu
            src/cuda/dense_compaction.cu
            src/cuda/dense_isolated_components.cu
            src/cuda/frame_execution.cu
            src/cuda/frame_overlap_scheduler.cu
            src/cuda/multi_frame_pipeline.cu
            src/cuda/overlap_debug.cu
            src/cuda/pipeline_debug.cu
            src/cuda/pipeline_workspace.cu
            src/cuda/scan_vertices.cu
            src/cuda/scan_projection.cu
            src/cuda/scan_range_fill.cu
            src/cuda/scan_vertex_build.cu
            src/cuda/ri_thin_filter.cu
            src/cuda/stitch_debug.cpp
            src/cuda/stitch_connection_mask.cu
            src/cuda/stitch_append.cu
            src/cuda/stitch_transaction.cu
            src/cuda/stitch_compaction.cpp
            src/cuda/stitch_geometry.cu
            src/cuda/stitch_gpu_helpers.cu
            src/cuda/stitch_proxy_cpu.cpp
            src/cuda/stitch_spatial.cpp
    )
    target_compile_definitions(ri_meshing_core PUBLIC RI_MESHING_HAS_CUDA=1)
    target_compile_options(ri_meshing_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
    target_compile_options(ri_meshing_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

add_executable(ri_meshing main.cpp)
target_link_libraries(ri_meshing PRIVATE ri_meshing_core)

if(RI_MESHING_BUILD_TESTS)
    enable_testing()
    add_executable(config_validation_tests tests/config_validation_tests.cpp)
    target_link_libraries(config_validation_tests PRIVATE ri_meshing_core)
    add_test(NAME config_validation_tests COMMAND config_validation_tests)

    add_executable(geometry_tests tests/geometry_tests.cpp)
    target_link_libraries(geometry_tests PRIVATE ri_meshing_core)
    add_test(NAME geometry_tests COMMAND geometry_tests)

    add_executable(stitch_candidates_tests tests/stitch_candidates_tests.cpp)
    target_include_directories(stitch_candidates_tests PRIVATE src)
    target_link_libraries(stitch_candidates_tests PRIVATE ri_meshing_core)
    add_test(NAME stitch_candidates_tests COMMAND stitch_candidates_tests)

    add_executable(stitch_boundary_filters_tests tests/stitch_boundary_filters_tests.cpp)
    target_include_directories(stitch_boundary_filters_tests PRIVATE src)
    target_link_libraries(stitch_boundary_filters_tests PRIVATE ri_meshing_core)
    add_test(NAME stitch_boundary_filters_tests COMMAND stitch_boundary_filters_tests)

    add_executable(stitch_matching_tests tests/stitch_matching_tests.cpp)
    target_include_directories(stitch_matching_tests PRIVATE src)
    target_link_libraries(stitch_matching_tests PRIVATE ri_meshing_core)
    add_test(NAME stitch_matching_tests COMMAND stitch_matching_tests)

    add_executable(stitch_connection_tests tests/stitch_connection_tests.cpp)
    target_include_directories(stitch_connection_tests PRIVATE src)
    target_link_libraries(stitch_connection_tests PRIVATE ri_meshing_core)
    add_test(NAME stitch_connection_tests COMMAND stitch_connection_tests)

    add_executable(hole_filling_tests tests/hole_filling_tests.cpp)
    target_link_libraries(hole_filling_tests PRIVATE ri_meshing_core)
    add_test(NAME hole_filling_tests COMMAND hole_filling_tests)

    add_executable(io_tests tests/io_tests.cpp)
    target_link_libraries(io_tests PRIVATE ri_meshing_core)
    add_test(NAME io_tests COMMAND io_tests)

    if(RI_MESHING_ENABLE_CUDA)
        set_source_files_properties(
            tests/ri_range_filter_cuda_tests.cpp
            tests/ri_thin_filter_cuda_tests.cpp
            PROPERTIES LANGUAGE CUDA
        )
        add_executable(ri_range_filter_cuda_tests tests/ri_range_filter_cuda_tests.cpp)
        target_include_directories(ri_range_filter_cuda_tests PRIVATE src/cuda)
        target_link_libraries(ri_range_filter_cuda_tests PRIVATE ri_meshing_core)
        add_test(NAME ri_range_filter_cuda_tests COMMAND ri_range_filter_cuda_tests)

        add_executable(ri_thin_filter_cuda_tests tests/ri_thin_filter_cuda_tests.cpp)
        target_include_directories(ri_thin_filter_cuda_tests PRIVATE src/cuda)
        target_link_libraries(ri_thin_filter_cuda_tests PRIVATE ri_meshing_core)
        add_test(NAME ri_thin_filter_cuda_tests COMMAND ri_thin_filter_cuda_tests)

        set_source_files_properties(
            tests/accumulated_depth_culling_tests.cpp
            PROPERTIES LANGUAGE CUDA
        )
        add_executable(accumulated_depth_culling_tests tests/accumulated_depth_culling_tests.cpp)
        target_include_directories(accumulated_depth_culling_tests PRIVATE src/cuda)
        target_link_libraries(accumulated_depth_culling_tests PRIVATE ri_meshing_core)
        add_test(NAME accumulated_depth_culling_tests COMMAND accumulated_depth_culling_tests)

    endif()

endif()
