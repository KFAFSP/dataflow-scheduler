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
// This is a tool for consuming KTIR mlir and generating KTDF and/or DFIR.
//===----------------------------------------------------------------------===//

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <tools/scheduler-driver-lib/scheduler-main.h>

#include "Dialects/Includes/init-all-dialects.h"
#include "Transform/Passes.h"

int main(int argc, char** argv) {
  registerSchedulerTransformsPasses();
  scheduler::registerPassPipelinesForScheduler();
  mlir::DialectRegistry registry;
  scheduler::registerAllDialects(registry);
  return failed(scheduler::SchedulerOptMain(
      argc, argv, "KTIR Scheduler modular optimizer driver", registry));
}
