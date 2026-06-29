//===-- dataflow-scheduler-main.h -------------------------------*- c++ -*-===//
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

#ifndef DATAFLOWSCHEDULERMAIN_H
#define DATAFLOWSCHEDULERMAIN_H

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
class MLIRContext;
}  // end namespace mlir

using namespace mlir;
namespace scheduler {

/// Configuration options for the scheduler tool.
class SchedulerOptMainConfig {
 public:
  static void registerCLOptions(DialectRegistry& dialectRegistry);
  static SchedulerOptMainConfig& createFromCLOptions();

  SchedulerOptMainConfig& allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  SchedulerOptMainConfig& setDebugConfig(tracing::DebugConfig config) {
    debugConfig = std::move(config);
    return *this;
  }
  tracing::DebugConfig& getDebugConfig() { return debugConfig; }
  const tracing::DebugConfig& getDebugConfig() const { return debugConfig; }

  SchedulerOptMainConfig& dumpPassPipeline(bool dump) {
    dumpPassPipelineFlag = dump;
    return *this;
  }
  bool shouldDumpPassPipeline() const { return dumpPassPipelineFlag; }

  SchedulerOptMainConfig& emitBytecode(bool emit) {
    emitBytecodeFlag = emit;
    return *this;
  }
  bool shouldEmitBytecode() const { return emitBytecodeFlag; }

  SchedulerOptMainConfig& setPassPipelineSetupFn(
      std::function<LogicalResult(PassManager&)> callback) {
    passPipelineCallback = std::move(callback);
    return *this;
  }

  SchedulerOptMainConfig& setPassPipelineParser(
      const PassPipelineCLParser& parser);

  LogicalResult setupPassPipeline(PassManager& pm) const {
    if (passPipelineCallback) return passPipelineCallback(pm);
    return success();
  }

  SchedulerOptMainConfig& showDialects(bool show) {
    showDialectsFlag = show;
    return *this;
  }
  bool shouldShowDialects() const { return showDialectsFlag; }

  SchedulerOptMainConfig& splitInputFile(
      std::string splitMarker = kDefaultSplitMarker) {
    splitInputFileFlag = std::move(splitMarker);
    return *this;
  }
  StringRef inputSplitMarker() const { return splitInputFileFlag; }

  SchedulerOptMainConfig& outputSplitMarker(
      std::string splitMarker = kDefaultSplitMarker) {
    outputSplitMarkerFlag = std::move(splitMarker);
    return *this;
  }
  StringRef getOutputSplitMarkerFlag() const { return outputSplitMarkerFlag; }

  SchedulerOptMainConfig& verifyPasses(bool verify) {
    verifyPassesFlag = verify;
    return *this;
  }
  bool shouldVerifyPasses() const { return verifyPassesFlag; }

  bool shouldPrintFinalIr() const { return printFinalIRFlag; }

  std::string getInputFileName() const { return inputFileName; }
  std::string getOutputFileName() const { return outputFileName; }

 protected:
  std::string inputFileName = "-";
  std::string outputFileName = "-";
  bool allowUnregisteredDialectsFlag = false;
  tracing::DebugConfig debugConfig;
  bool dumpPassPipelineFlag = false;
  bool emitBytecodeFlag = false;
  std::function<LogicalResult(PassManager&)> passPipelineCallback;
  bool showDialectsFlag = false;
  std::string splitInputFileFlag = "";
  std::string outputSplitMarkerFlag = kDefaultSplitMarker;
  bool verifyPassesFlag = true;
  bool printFinalIRFlag = true;
};

/// Perform the core processing behind `scheduler`.
LogicalResult SchedulerOptMain(llvm::raw_ostream& outputStream,
                               std::unique_ptr<llvm::MemoryBuffer> buffer,
                               mlir::DialectRegistry& registry,
                               const SchedulerOptMainConfig& config);

/// Implementation for tools like `scheduler`.
LogicalResult SchedulerOptMain(int argc, char** argv, llvm::StringRef toolName,
                               DialectRegistry& registry);

void registerPassesForScheduler();
void registerPassPipelinesForScheduler();
void registerAndParseCLIOptions(int argc, char** argv, llvm::StringRef toolName,
                                DialectRegistry& registry);

}  // end namespace scheduler

#endif  // DATAFLOWSCHEDULERMAIN_H
