//===-------------------- SchedulerExtContext.cpp -------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/Utils/SchedulerExtContext.h"

#include <cstring>

#include "mlir/IR/Builders.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

const SchedulerExtContext& SchedulerExtContext::dummyContext() {
  // The dummy context is initialized thread-safe and never written to.
  static const DummySchedulerExtContext dummy_ctx;
  return dummy_ctx;
}

uint64_t SchedulerExtContext::getStickSize() const {
  return 64;  // 64 bytes per stick
}

mlir::StringAttr SchedulerExtContext::getTransferMemSpace(
    mlir::MLIRContext* context, const char* component) const {
  // Map component types to memory space names
  if (std::strcmp(component, "SFP") == 0) {
    return mlir::StringAttr::get(context, "SFP_REG");
  }
  // Default to DRAM if component not recognized
  return mlir::StringAttr::get(context, "DRAM");
}

mlir::ktdf::FifoType SchedulerExtContext::getFifoTypeLoad() const {
  // Load operations transfer data from LXLU to SFP
  return mlir::ktdf::FifoType::LxluToSfp;
}

mlir::ktdf::FifoType SchedulerExtContext::getFifoTypeStore() const {
  // Store operations transfer data from SFP to LXSU
  return mlir::ktdf::FifoType::SfpToLxsu;
}

}  // namespace scheduler
