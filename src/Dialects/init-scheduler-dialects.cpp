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
#include "src/Dialects/Includes/init-scheduler-dialects.h"

#include "mlir/IR/DialectRegistry.h"
#include "src/Dialects/Includes/scheduler-dialects.h"
#include "src/Transform/Passes.h"

namespace scheduler {

void registerSchedulerDialects(mlir::DialectRegistry& registry) {
  // clang-format off
  registry.insert<mlir::ktdf::KTDFDialect>();
  registry.insert<mlir::agen::AgenDialect>();
  registry.insert<mlir::dataflow::DataflowDialect>();
  registry.insert<mlir::uniform::UniformDialect>();
  registry.insert<mlir::vectorchain::VectorChainDialect>();
  // Future scheduler dialects can be added here
  // clang-format on
}

void registerAllSchedulerPasses() { registerSchedulerTransformPasses(); }

}  // namespace scheduler
