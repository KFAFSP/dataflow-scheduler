//===-- Pipeline.h ----------------------------------------------*- c++ -*-===//
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
// This file declares the scheduler pipelie.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_PIPELINE_H_
#define DATAFLOW_SCHEDULER_PIPELINE_H_

#include <llvm/ADT/StringRef.h>

namespace mlir {

class OpPassManager;

}  // namespace mlir

namespace scheduler {

struct SchedulerExtContext;

/// Builds the KTDP to DFIR pipeline with the given pass manager.
void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx,
    llvm::StringRef split_output_dir);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_PIPELINE_H_
