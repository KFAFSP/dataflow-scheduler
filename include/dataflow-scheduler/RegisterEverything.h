//===-- RegisterEverything.h ------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
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
//
// This file declares the main registration entry point for the scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_REGISTEREVERYTHING_H_
#define DATAFLOW_SCHEDULER_REGISTEREVERYTHING_H_

namespace mlir {

class DialectRegistry;

}  // namespace mlir

namespace scheduler {

//===----------------------------------------------------------------------===//
// Exported Only
//===----------------------------------------------------------------------===//

/// Registers all passed defined by the scheduler.
void registerPasses();
/// Registers all dialects defined by the scheduler.
void registerDialects(mlir::DialectRegistry& registry);
/// Registers all extensions provided by the scheduler.
void registerExtensions(mlir::DialectRegistry& registry);

//===----------------------------------------------------------------------===//
// Imported and Exported
//===----------------------------------------------------------------------===//

/// Registers all passes defined and used by the scheduler.
void registerAllPasses();
/// Registers all dialects defined and used by the scheduler.
void registerAllDialects(mlir::DialectRegistry& registry);
/// Registers all extensions provided and required by the scheduler.
void registerAllExtensions(mlir::DialectRegistry& registry);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_REGISTEREVERYTHING_H_
