//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

#ifndef SCHEDULERMAIN_H
#define SCHEDULERMAIN_H

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Debug/Counter.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/Observers/ActionLogging.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace llvm {
class raw_ostream;
class MemoryBuffer;
}  // end namespace llvm

namespace mlir {
class DialectRegistry;
class PassPipelineCLParser;
class PassManager;
}  // end namespace mlir

using namespace mlir;
namespace scheduler {

/// Configuration options for the scheduler tool.
/// This is based on KtirOptMainConfig from KTIR.
class SchedulerOptMainConfig {
 public:
  /// Register the options as global LLVM command line options.
  static void registerCLOptions(DialectRegistry &dialectRegistry);

  /// Create a new config with the default set from the CL options.
  static SchedulerOptMainConfig &createFromCLOptions();

  ///
  /// Options.
  ///

  /// Allow operation with no registered dialects.
  SchedulerOptMainConfig &allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  /// Set the debug configuration to use.
  SchedulerOptMainConfig &setDebugConfig(tracing::DebugConfig config) {
    debugConfig = std::move(config);
    return *this;
  }
  tracing::DebugConfig &getDebugConfig() { return debugConfig; }
  const tracing::DebugConfig &getDebugConfig() const { return debugConfig; }

  /// Print the pass-pipeline as text before executing.
  SchedulerOptMainConfig &dumpPassPipeline(bool dump) {
    dumpPassPipelineFlag = dump;
    return *this;
  }
  bool shouldDumpPassPipeline() const { return dumpPassPipelineFlag; }

  /// Set the output format to bytecode instead of textual IR.
  SchedulerOptMainConfig &emitBytecode(bool emit) {
    emitBytecodeFlag = emit;
    return *this;
  }
  bool shouldEmitBytecode() const { return emitBytecodeFlag; }

  /// Set the callback to populate the pass manager.
  SchedulerOptMainConfig &setPassPipelineSetupFn(
      std::function<LogicalResult(PassManager &)> callback) {
    passPipelineCallback = std::move(callback);
    return *this;
  }

  /// Set the parser to use to populate the pass manager.
  SchedulerOptMainConfig &setPassPipelineParser(
      const PassPipelineCLParser &parser);

  /// Populate the passmanager, if any callback was set.
  LogicalResult setupPassPipeline(PassManager &pm) const {
    if (passPipelineCallback) return passPipelineCallback(pm);
    return success();
  }

  /// Show the registered dialects before trying to load the input file.
  SchedulerOptMainConfig &showDialects(bool show) {
    showDialectsFlag = show;
    return *this;
  }
  bool shouldShowDialects() const { return showDialectsFlag; }

  /// Set whether to run the verifier after each transformation pass.
  SchedulerOptMainConfig &verifyPasses(bool verify) {
    verifyPassesFlag = verify;
    return *this;
  }
  bool shouldVerifyPasses() const { return verifyPassesFlag; }

  bool shouldPrintFinalIr() const { return printFinalIRFlag; }

 protected:
  /// Allow operation with no registered dialects.
  bool allowUnregisteredDialectsFlag = false;

  /// Configuration for the debugging hooks.
  tracing::DebugConfig debugConfig;

  /// Print the pipeline that will be run.
  bool dumpPassPipelineFlag = false;

  /// Emit bytecode instead of textual assembly when generating output.
  bool emitBytecodeFlag = false;

  /// The callback to populate the pass manager.
  std::function<LogicalResult(PassManager &)> passPipelineCallback;

  /// Show the registered dialects before trying to load the input file.
  bool showDialectsFlag = false;

  /// Run the verifier after each transformation pass.
  bool verifyPassesFlag = true;

  /// Print final ir after all transformation passes
  bool printFinalIRFlag = true;
};

/// Perform the core processing behind `scheduler`.
LogicalResult SchedulerOptMain(llvm::raw_ostream &outputStream,
                               std::unique_ptr<llvm::MemoryBuffer> buffer,
                               mlir::DialectRegistry &registry,
                               const SchedulerOptMainConfig &config);

/// Implementation for tools like `scheduler`.
LogicalResult SchedulerOptMain(int argc, char **argv, llvm::StringRef toolName,
                               DialectRegistry &registry);

void registerPassesForScheduler();
void registerPassPipelinesForScheduler();

}  // end namespace scheduler

#endif  // SCHEDULERMAIN_H
