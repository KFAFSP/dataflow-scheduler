#!/bin/bash

#
# Copyright 2026 The KTIR Scheduler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Set your source root, or let it be inferred from the script location:
: "${KTIR_SCHEDULER_SRC=$(dirname "$0")/..}"

# Set your LLVM build folder or install prefix, or adjust individually:
: "${LLVM_INSTALL_PREFIX:=~/workspaces/llvm-project-v22.1/build}"
: "${LLVM_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/llvm}"
: "${LLVM_EXTERNAL_LIT=$LLVM_INSTALL_PREFIX/bin/llvm-lit}"
: "${MLIR_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/mlir}"

# Step 1: Configure.
cmake -S $KTIR_SCHEDULER_SRC -B $KTIR_SCHEDULER_SRC/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_INSTALL_PREFIX=$KTIR_SCHEDULER_SRC/install \
      -DLLVM_DIR=$LLVM_DIR \
      -DLLVM_EXTERNAL_LIT=$LLVM_EXTERNAL_LIT \
      -DMLIR_DIR=$MLIR_DIR
      # To debug linking issues, add:
      # -DBUILD_SHARED_LIBS=ON
      # To build the Python bindings, add:
      # -DKTIR_ENABLE_PYTHON_BINDINGS=ON

# Step 2: Build.
cmake --build $KTIR_SCHEDULER_SRC/build --target all
