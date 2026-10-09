# RIMBA

RIMBA is a C++23 and CUDA implementation of sparse range-image meshing for
sequential LiDAR scans. It builds a mesh from successive range images, with
CUDA-based per-frame processing, boundary stitching, and configurable overlap
scheduling.

This repository contains the algorithm implementation, public C++ headers,
command-line executable, and tests. Experiment protocols and analysis tools are
maintained separately in [RIMBA Experiments](https://github.com/rimba-meshing/rimba-experiments).

## Project structure

- `include/ri_meshing/`: public C++ headers and configuration types.
- `src/`: CPU-side implementation and shared pipeline code.
- `src/cuda/`: CUDA kernels and GPU pipeline implementation.
- `tests/`: unit and CUDA tests.
- `main.cpp`: command-line application that reads a sequence and writes a PLY mesh.

The CMake library target is `ri_meshing_core`; the executable is `ri_meshing`.
The project currently builds from source and does not provide an install or
package-manager workflow.

## Requirements

- CMake 4.2 or newer.
- A C++23-compatible compiler.
- CUDA Toolkit with `nvcc` and a supported NVIDIA GPU for meshing runs.
- OpenMP is enabled when available; it is optional for configuration.

The CPU presets build shared utilities and CPU tests. The meshing executable
requires a CUDA build. Python is not required by the library; the experiments
repository uses Python 3.10 or newer for protocol orchestration and analysis.

## Build and test

### CPU utilities and tests

Use this preset for CPU-only development and shared utility tests:

```bash
cmake --preset dev
cmake --build --preset dev -j"$(nproc)"
ctest --preset dev
```

The `release` preset provides an optimized CPU build with tests enabled.

### CUDA development build

This Debug preset builds the GPU pipeline and CUDA tests:

```bash
cmake --preset cuda
cmake --build --preset cuda -j"$(nproc)"
ctest --preset cuda
```

### Optimized CUDA build

For a Release build targeting the CUDA architecture detected on the current
host:

```bash
cmake --preset cuda-release-native
cmake --build --preset cuda-release-native -j"$(nproc)"
ctest --preset cuda-release-native
```

The generated executable is `build/cuda-native-release/ri_meshing`.
`CMAKE_CUDA_ARCHITECTURES=native` makes this preset specific to the build host;
select an explicit architecture when building for a different GPU.

## Run the meshing executable

The command-line application reads KITTI-style binary point clouds and pose
files from a sequence directory. The input root is expected to contain files
in this layout:

```text
<dataset-root>/
└── sequences/<sequence>/
    ├── velodyne/000000.bin
    ├── poses.txt
    └── calib.txt
```

Each `.bin` file stores four `float` values per point, as in KITTI odometry
data; the fourth value is ignored. The pose and calibration files are read from
the sequence directory. The optional `--use-custom-poses` flag selects
`poses_custom.txt` instead of `poses.txt`.

Example: mesh ten frames beginning at frame zero from sequence `07`:

```bash
./build/cuda-native-release/ri_meshing \
  --dataset /path/to/dataset \
  --sequence 07 \
  --frame 0 \
  --frame-count 10 \
  --output build/meshes/sequence-07.ply
```

Run `ri_meshing --help` to see the available options. They include frame range,
warm-up and measured runs, overlap mode, hole filling, timing detail, and
optional pipeline diagnostics. The command writes an ASCII PLY mesh to the
path selected with `--output`.

## Reproducible experiments

The source revision used for a published or reported measurement should be
recorded by the corresponding commit in
[RIMBA Experiments](https://github.com/rimba-meshing/rimba-experiments). That
repository pins its selected source revision as a Git submodule and documents
the datasets, build variants, benchmark boundaries, and analysis procedure in
its [reproducibility guide](https://github.com/rimba-meshing/rimba-experiments/blob/main/REPRODUCIBILITY.md).

## Citation and distribution

RIMBA is research software under active development. Citation and license
information will be added when the associated publication and release terms
are finalized.
