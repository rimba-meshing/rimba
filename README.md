# RIMBA

RIMBA is a C++/CUDA implementation of sparse range-image meshing. This repository contains the library implementation, command-line executable, public headers, and tests.

## Requirements

- CMake 4.2 or newer
- A C++23 compiler for CPU builds
- A CUDA toolkit and `nvcc` for CUDA builds

## CPU build

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

## CUDA build and tests

```bash
cmake --preset cuda
cmake --build --preset cuda
ctest --preset cuda
```

The public C++ headers are under `include/ri_meshing/`. The current CMake
library and executable targets retain the `ri_meshing` names from the source
implementation.
