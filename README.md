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

3. **KTIR (ktir-mlir-frontend)** - The KTIR Data Plane dialect and infrastructure
   ```bash
   git clone https://github.com/torch-spyre/ktir-mlir-frontend.git
   cd ktir-mlir-frontend
   mkdir build && cd build
   cmake -G Ninja .. \
     -DCMAKE_BUILD_TYPE=Debug \
     -DLLVM_PROJ_SRC=<path-to-llvm-project> \
     -DLLVM_PROJ_BUILD=<path-to-llvm-project>/build
   ninja
   ```

## Build Instructions

### Step 1: Clone KTIR-Scheduler

```bash
git clone git@github.ibm.com:ai-chip-toolchain/KTIR-Scheduler.git
cd KTIR-Scheduler
```

### Step 2: Configure Build Paths

Edit `scripts/buildKTIRScheduler.sh` to set the correct paths for your environment:
- `LLVM_PROJ_SRC`: Path to your LLVM source directory
- `LLVM_PROJ_BUILD`: Path to your LLVM build directory
- `KTIR_PATH`: Path to your ktir-mlir-frontend source directory
- `KTIR_BUILD_PATH`: Path to your ktir-mlir-frontend build directory

### Step 3: Build Using the Build Script

```bash
mkdir -p build
./scripts/buildKTIRScheduler.sh
```

### Step 4: Manual Build (Alternative)

Alternatively, you can build manually with CMake:

```bash
mkdir -p build
cd build

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DLLVM_PROJ_SRC=<path-to-llvm-project> \
  -DLLVM_PROJ_BUILD=<path-to-llvm-project>/build \
  -DKTIR_PROJ_SRC=<path-to-ktir-mlir-frontend> \
  -DKTIR_PROJ_BUILD=<path-to-ktir-mlir-frontend>/build \
  -DCMAKE_INSTALL_PREFIX=../install

ninja
```

The build will produce the `scheduler` executable in `build/scheduler/bin/`.

## Running Tests

To run the test suite:

```bash
cd build
ninja check-scheduler
```

This will execute all tests defined in the `test/` directory using the LLVM lit testing framework.

