//===-- DataPath.cpp ---------------------------------------------*- c++-*-===//
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
// This file implements the data path analysis.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/DataPath.h"

using namespace mlir;
using namespace mlir::ktdf;

auto mlir::ktdf::traceDataPath(Value generic_input, PipelineOp pipeline)
    -> std::optional<DataPath> {
  DataPath path;
  path.target = generic_input;

  // Step 1: generic_input must come from read_from_fifo
  auto read_op = generic_input.getDefiningOp<ReadFromFifoOp>();
  if (!read_op) return std::nullopt;

  Value fifo_slot = read_op.getFifoSlot();

  // Step 2: Trace backward through hops until we reach the DDR source.
  Value current_value = fifo_slot;

  while (true) {
    // Find the data_transfer inside the pipeline whose destination is
    // current_value (either directly or via ktdf.private result).
    DataTransferOp found_transfer;
    pipeline.walk([&](DataTransferOp transfer) {
      if (found_transfer) return WalkResult::interrupt();
      Value dest = transfer.getDestination();
      if (dest == current_value) {
        found_transfer = transfer;
        return WalkResult::interrupt();
      }
      // Also match if dest is a ktdf.private result that yields current_value
      if (auto priv = dest.getDefiningOp<PrivateOp>()) {
        auto result_idx = cast<OpResult>(dest).getResultNumber();
        PrivateYieldOp yield = priv.getYieldOp();
        if (yield && result_idx < yield.getNumOperands() &&
            yield.getOperand(result_idx) == current_value) {
          found_transfer = transfer;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (!found_transfer) return std::nullopt;

    DataPathHop hop;
    hop.transfer = found_transfer;
    hop.private_resource = current_value;
    path.hops.push_back(hop);

    Value src = found_transfer.getSource();

    // If source is not from ktdf.private, it's the DDR origin — done.
    if (!src.getDefiningOp<PrivateOp>()) {
      path.source = src;
      break;
    }

    // Source is a private resource; continue tracing backward.
    current_value = src;
  }

  // Reverse so hops are in source→target order.
  std::reverse(path.hops.begin(), path.hops.end());
  return path;
}
