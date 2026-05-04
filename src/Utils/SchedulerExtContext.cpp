//===-------------------- SchedulerExtContext.cpp -------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/Utils/SchedulerExtContext.h"

namespace scheduler {

const SchedulerExtContext& SchedulerExtContext::dummyContext() {
  // The dummy context is initialized thread-safe and never written to.
  static const DummySchedulerExtContext dummy_ctx;
  return dummy_ctx;
}

uint64_t SchedulerExtContext::getStickSize() const {
  return 64;  // 64 bytes per stick
}

}  // namespace scheduler
