//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#ifndef SCHEDULER_EXT_CONTEXT_H
#define SCHEDULER_EXT_CONTEXT_H

#include <cstdint>

namespace scheduler {

struct SchedulerExtContext {
  SchedulerExtContext() {}

  /// @brief Get a default initialized context for scheduler.
  /// The dummy context is intended for use with passes and analyses that
  /// require default constructors.
  static const SchedulerExtContext& dummyContext();

  virtual bool isDummy() const { return false; }

  /// Returns the size of a stick in bytes (64 bytes for current architecture)
  uint64_t getStickSize() const;
};

/// @brief Construct a DummySchedulerExtContext. In general prefer
/// SchedulerExtContext::dummyContext.
struct DummySchedulerExtContext : SchedulerExtContext {
  DummySchedulerExtContext() : SchedulerExtContext() {}
  bool isDummy() const override { return true; }
};

}  // namespace scheduler

#endif  // SCHEDULER_EXT_CONTEXT_H