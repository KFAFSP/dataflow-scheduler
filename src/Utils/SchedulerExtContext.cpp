//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/Utils/SchedulerExtContext.h"

#include "mlir/IR/Builders.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

const SchedulerExtContext& SchedulerExtContext::dummyContext() {
  // The dummy context is initialized thread-safe and never written to.
  static const DummySchedulerExtContext dummy_ctx;
  return dummy_ctx;
}

mlir::ktdf::ComponentType SchedulerExtContext::getComputeUnit() const {
  return mlir::ktdf::ComponentType::SFP;
}

uint64_t SchedulerExtContext::getVectorLength(mlir::ktdf::ComponentType unit,
                                              mlir::Type elem_type) const {
  if (unit == mlir::ktdf::ComponentType::SFP && elem_type.isF16()) {
    return 64;
  }
  llvm::report_fatal_error(
      "Compute unit and precision combination not supported");
}

uint64_t SchedulerExtContext::getMaxFifoSlotSize(
    mlir::ktdf::ComponentType unit) const {
  if (unit == mlir::ktdf::ComponentType::SFP) {
    return 64;
  }
  llvm::report_fatal_error("Compute unit not supported");
}

std::string SchedulerExtContext::getInterfaceMemToComputeUnit(
    mlir::ktdf::ComponentType unit) const {
  if (unit == mlir::ktdf::ComponentType::SFP) {
    return "FIFO";
  }
  return "MEM";
}

mlir::ktdf::FifoType SchedulerExtContext::getFifoTypeLoad() const {
  // Load operations transfer data from HBM to SFP
  return mlir::ktdf::FifoType::HbmToSfp;
}

mlir::ktdf::FifoType SchedulerExtContext::getFifoTypeStore() const {
  // Store operations transfer data from SFP to HBM
  return mlir::ktdf::FifoType::SfpToHbm;
}

}  // namespace scheduler
