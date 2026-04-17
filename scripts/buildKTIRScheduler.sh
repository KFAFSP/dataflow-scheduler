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

KTIR_SCHEDULER_PATH=~/workspaces/KTIR-Scheduler/
LLVM_PROJ_SRC=~/workspaces/llvm-project
LLVM_PROJ_BUILD=~/workspaces/llvm-project/build/
KTIR_PROJ_SRC=~/workspaces/ktir-mlir-frontend/
KTIR_PROJ_BUILD=$KTIR_PATH/build/

cd $KTIR_SCHEDULER_PATH/build

cmake -G Ninja $KTIR_SCHEDULER_PATH -DCMAKE_BUILD_TYPE=Debug  \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      -DLLVM_PROJ_SRC=$LLVM_PROJ_SRC \
      -DLLVM_PROJ_BUILD=$LLVM_PROJ_BUILD \
      -DKTIR_PROJ_SRC=$KTIR_PROJ_SRC \
      -DKTIR_PROJ_BUILD=$KTIR_PROJ_BUILD \
      -DCMAKE_INSTALL_PREFIX=$KTIR_SCHEDULER_PATH/install
