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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/MappableExtensions.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/OperationSupport.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

struct StageModel : ktdf_arch::Mappable::ExternalModel<StageModel, StageOp> {
  static auto getMapsTo(Operation* op) -> mlir::ktdf_arch::MapsToAttr {
    return dyn_cast<mlir::ktdf_arch::MapsToAttr>(
        cast<StageOp>(op).getApplicableUnitsAttr());
  }

  static auto setMapsTo(Operation* op, mlir::ktdf_arch::MapsToAttr maps_to)
      -> LogicalResult {
    auto units = dyn_cast<ArrayAttr>(maps_to);
    if (!units) {
      units = ArrayAttr::get(maps_to.getContext(), {maps_to});
    }

    cast<StageOp>(op).setApplicableUnitsAttr(units);
    return success();
  }

  static auto removeMapsTo(Operation* op) -> mlir::ktdf_arch::MapsToAttr {
    return dyn_cast<mlir::ktdf_arch::MapsToAttr>(
        cast<StageOp>(op).removeApplicableUnitsAttr());
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

struct PipelineModel
    : ktdf_arch::Mappable::ExternalModel<PipelineModel, PipelineOp> {
  static auto getMapsTo(Operation* op) -> ktdf_arch::MapsToAttr {
    SmallVector<ktdf_arch::ResourceSpecAttr> result;
    for (auto stage : cast<PipelineOp>(op).getStages()) {
      const auto units = StageModel::getMapsTo(stage);
      if (!units) {
        return nullptr;
      }

      llvm::append_range(result, units.getValue());
    }

    return ktdf_arch::MapsToAttr::get(op->getContext(), result);
  }

  static auto setMapsTo(Operation* /*op*/,
                        mlir::ktdf_arch::MapsToAttr /*maps_to*/)
      -> LogicalResult {
    return failure();
  }

  static auto removeMapsTo(Operation* op) -> ktdf_arch::MapsToAttr {
    SmallVector<ktdf_arch::ResourceSpecAttr> result;
    for (auto stage : cast<PipelineOp>(op).getStages()) {
      const auto units = StageModel::removeMapsTo(stage);
      if (!units) {
        continue;
      }

      llvm::append_range(result, units.getValue());
    }

    return ktdf_arch::MapsToAttr::get(op->getContext(), result);
  }
};

}  // namespace

void mlir::ktdf::registerMappableInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, KTDFDialect* /*dialect*/) {
    StageOp::attachInterface<StageModel>(*ctx);
    PipelineOp::attachInterface<PipelineModel>(*ctx);
  });
}
