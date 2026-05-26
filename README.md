# KTIR-Scheduler

## Description

KTIR-Scheduler is a compiler infrastructure for transforming KTIR programs into optimized, pipelined execution schedules for specialized dataflow hardware accelerators. The scheduler takes a KTIR program—a single module containing computation expressed in the Kernel Tiled Data Parallel (KTDP) abstraction—and transforms it into a Kernel Tiled Data Flow (KTDF) representation with explicit pipeline stages, memory hierarchy management, and high-level optimizations based on hardware-specific configurations.

The transformation pipeline partitions computations into compute groups, constructs multi-stage pipelines with explicit data movement through the memory hierarchy (eg. DRAM → LX scratchpad → compute registers), and applies optimizations such as tiling and other loop transformations to expose parallelism, exploit data reuse, and generate efficient schedules that maximize hardware utilization while respecting architectural constraints.

## Prerequisites

Before building KTIR-Scheduler, you need the following dependencies:

1. **Build Tools**
   - CMake (version 3.20.0 or higher)
   - Ninja build system
   - Clang/Clang++ compiler
   - Git

2. **LLVM Project** - The LLVM/MLIR infrastructure
   ```bash
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   mkdir build && cd build
   cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS=mlir \
     -DLLVM_TARGETS_TO_BUILD="host" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON
   ninja
   ```

## Build Instructions

Building `KTIR-scheduler` automatically fetches and builds the required KTIR submodules, but requires a working build or installation of LLVM and MLIR to be present on your system.

```sh
# Clone `KTIR-scheduler` and all its submodules.
git clone --recursive git@github.ibm.com:ai-chip-toolchain/KTIR-Scheduler.git
cd KTIR-Scheduler

# Configure the project.
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR=$LLVM_INSTALL_PREFIX/$lib/cmake/llvm \
  -DMLIR_DIR=$MLIR_INSTALL_PREFIX/$lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit

# Build everything.
cmake --build build --target all
```

The build will produce the `scheduler` executable in `build/bin/`.

The following CMake variables can be configured:

|              Name | Type      | Description |
| ----------------: | :-------- | --- |
| `LLVM_DIR` <br/>*(required)* | `STRING` | Path to the CMake directory of an **LLVM** installation. <br/> *e.g. `~/tools/llvm-12.0.1/lib/cmake/llvm`* |
| `MLIR_DIR` <br/>*(required)* | `STRING` | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `~/tools/llvm-12.0.1/lib/cmake/mlir`* |
| `LLVM_EXTERNAL_LIT` <br/>*(optional)* | `STRING` | Path to an `llvm-lit` executable, required for testing. |
| `KTIR_ENABLE_PYTHON_BINDINGS` <br/>*(default: `OFF`)* | `BOOL` | Whether to build the Python bindings. |
| `KTIR_BUILD_TOOLS` <br/>*(default: `ON`)* | `BOOL` | Whether to build the tool executables. |

## Running Tests

The entire set of regression tests can be run with the following command:

```sh
cmake --build build --target check-all
```

To test the external KTDP part, build `check-ktdp`. To test all dialects defined in the KTIR project, build `check-ktir`. To test the scheduler defined in this repository, build `check-scheduler`.

## Python Bindings

When `KTIR_ENABLE_PYTHON_BINDINGS` is set, the Python bindings will be built and output into `python_packages/ktir/mlir_ktir`, both in the build and install directories.

To test the Python bindings from the build directory, run:

```sh
cmake --build build --target check-scheduler-python
```

These tests are also included in the `check-all` target, if bindings are enabled. To test the Python bindings from the install directory instead, run:

```sh
cmake --build build --target install
PYTHONPATH=$CMAKE_INSTALL_PREFIX/python_packages/ktir python $KTIR_SRC_DIR/test/python/init.py
```
