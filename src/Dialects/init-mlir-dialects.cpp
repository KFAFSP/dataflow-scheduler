//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "src/Dialects/Includes/init-mlir-dialects.h"

#include "src/Dialects/Includes/mlir-dialects.h"
// #include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
// #include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"

using namespace mlir;
namespace scheduler {

void registerMLIRDialects(mlir::DialectRegistry& registry) {
  // clang-format off
  registry.insert<mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();

  // clang-format on

  // subset from mlir/include/mlir/InitAllDialects.h
  //  linalg::registerAllDialectInterfaceImplementations(registry);
  //  tensor::registerTilingInterfaceExternalModels(registry);
  //  mlir::func::registerInlinerExtension(registry);
}

}  // namespace scheduler
