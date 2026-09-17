//===-- MappableExtensions.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/Dataflow/Transforms/MappableExtensions.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

[[nodiscard]] auto getResource(GetUnitOp op) -> ktdf_arch::ResourceSpecAttr {
  return mlir::ktdf_arch::KindAttr(
      StringAttr::get(op.getContext(), op.getType().upper()));
}

struct ProgramUnitModel
    : ktdf_arch::Mappable::ExternalModel<ProgramUnitModel, ProgramUnitOp> {
  static auto getMapsTo(Operation* op) -> mlir::ktdf_arch::MapsToAttr {
    SmallVector<ktdf_arch::ResourceSpecAttr> result;
    for (auto unit : cast<ProgramUnitOp>(op).getUnits()) {
      const auto source = unit.getDefiningOp<GetUnitOp>();
      if (source == nullptr) {
        return nullptr;
      }

      result.push_back(getResource(source));
    }

    return ktdf_arch::MapsToAttr::get(op->getContext(), result);
  }

  static auto setMapsTo(Operation* /*op*/,
                        mlir::ktdf_arch::MapsToAttr /*maps_to*/)
      -> LogicalResult {
    // TODO: Implement this by updating the operands?
    return failure();
  }

  static auto removeMapsTo(Operation* /*op*/) -> mlir::ktdf_arch::MapsToAttr {
    return nullptr;
  }

  static auto verifyMapping(Operation* op,
                            ArrayRef<ktdf_arch::Resource> resources)
      -> LogicalResult {
    for (auto resource : resources) {
      if (!isa<ktdf_arch::ExecutionUnitOp>(resource)) {
        auto diag = op->emitError("invalid mapping: not an execution unit");
        diag.attachNote(resource->getLoc()) << "see this resource";
        return diag;
      }
    }

    return success();
  }
};

}  // namespace

void mlir::dataflow::registerMappableInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, DataflowDialect* /*dialect*/) {
    ProgramUnitOp::attachInterface<ProgramUnitModel>(*ctx);
  });
}
