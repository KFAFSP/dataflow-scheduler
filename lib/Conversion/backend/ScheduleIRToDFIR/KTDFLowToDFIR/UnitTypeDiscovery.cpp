//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"

using namespace scheduler;

namespace {

// Expand a single unit operand of a ktdf_lowering.execute_on into the
// underlying dataflow.get_unit Values it ranges over.
//
// If the operand is a uniform.query_map result, returns the values of its
// underlying uniform.def_immutable_mapping (which are dataflow.get_unit
// results, by Phase 1 construction).
//
// If the operand is a dataflow.get_unit result directly, returns that value
// as a single-element list.
//
// Returns failure if the operand can't be resolved to either form.
llvm::FailureOr<llvm::SmallVector<mlir::Value, 4>> expandToGetUnitValues(
    mlir::Value operand, mlir::Operation* diag_op) {
  llvm::SmallVector<mlir::Value, 4> out;
  mlir::Operation* def = operand.getDefiningOp();
  if (!def) {
    diag_op->emitError(
        "ktdflowering-to-dfir: could not resolve component type for unit "
        "operand of ktdf_lowering.execute_on");
    return mlir::failure();
  }

  if (mlir::isa<mlir::dataflow::GetUnitOp>(def)) {
    out.push_back(operand);
    return out;
  }

  if (auto query = mlir::dyn_cast<mlir::uniform::QueryMapOp>(def)) {
    mlir::Operation* map_def = query.getMap().getDefiningOp();
    auto def_map =
        map_def ? mlir::dyn_cast<mlir::uniform::DefImmutableMappingOp>(map_def)
                : nullptr;
    if (!def_map) {
      diag_op->emitError(
          "ktdflowering-to-dfir: could not resolve component type for unit "
          "operand of ktdf_lowering.execute_on");
      return mlir::failure();
    }
    for (mlir::Value v : def_map.getValues()) out.push_back(v);
    return out;
  }

  diag_op->emitError(
      "ktdflowering-to-dfir: could not resolve component type for unit "
      "operand of ktdf_lowering.execute_on");
  return mlir::failure();
}

}  // namespace

mlir::LogicalResult scheduler::discoverUnitTypes(
    llvm::ArrayRef<mlir::Operation*> work_ops, ResourceToUnits& result) {
  llvm::DenseMap<scheduler::ResourceType, llvm::DenseSet<mlir::Value>> seen;

  auto record = [&](scheduler::ResourceType rt, mlir::Value v) {
    auto& set = seen[rt];
    if (set.insert(v).second) {
      result[rt].push_back(v);
    }
  };

  for (mlir::Operation* root : work_ops) {
    mlir::WalkResult walk_result = root->walk([&](mlir::Operation* op)
                                                  -> mlir::WalkResult {
      if (auto exec = mlir::dyn_cast<mlir::ktdf_lowering::ExecuteOnOp>(op)) {
        for (mlir::Value u : exec.getUnits()) {
          auto rt_opt = scheduler::getUnitResourceType(u);
          if (!rt_opt) {
            exec.emitError(
                "ktdflowering-to-dfir: could not resolve component type "
                "for unit operand of ktdf_lowering.execute_on");
            return mlir::WalkResult::interrupt();
          }
          auto expanded = expandToGetUnitValues(u, exec.getOperation());
          if (mlir::failed(expanded)) return mlir::WalkResult::interrupt();
          for (mlir::Value v : *expanded) record(*rt_opt, v);
        }
      } else if (auto pu = mlir::dyn_cast<mlir::dataflow::ProgramUnitOp>(op)) {
        for (mlir::Value u : pu.getUnits()) {
          // For program_unit operands, get the resource directly from
          // the get_unit's "type" attribute.
          std::optional<scheduler::ResourceType> rt_opt;
          if (auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
                  u.getDefiningOp())) {
            auto type_attr = get_unit->getAttrOfType<mlir::StringAttr>("type");
            if (type_attr) {
              rt_opt = mlir::StringAttr::get(op->getContext(),
                                             type_attr.getValue().upper());
            }
          }

          if (!rt_opt) {
            pu.emitError(
                "ktdflowering-to-dfir: could not resolve component type "
                "for unit operand of dataflow.program_unit");
            return mlir::WalkResult::interrupt();
          }
          record(*rt_opt, u);
        }
      }
      return mlir::WalkResult::advance();
    });
    if (walk_result.wasInterrupted()) return mlir::failure();
  }
  return mlir::success();
}
