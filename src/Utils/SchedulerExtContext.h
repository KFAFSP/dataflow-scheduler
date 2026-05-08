//===-------------------------- SchedulerExtContext.h ---------------------===//
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

#include "mlir/IR/Attributes.h"
#include "src/Dialects/KTDF/KTDFTypes.hpp"

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

  /// Returns the memory space attribute for data transfers
  /// @param component The component type
  /// @return StringAttr with the memory space name (e.g., "SFP_REG")
  mlir::StringAttr getTransferMemSpace(mlir::MLIRContext* context,
                                       const char* component) const;

  /// Returns the FIFO type for load operations
  /// @return FifoType for loads (e.g., LxluToSfp)
  mlir::ktdf::FifoType getFifoTypeLoad() const;

  /// Returns the FIFO type for store operations
  /// @return FifoType for stores (e.g., SfpToLxsu)
  mlir::ktdf::FifoType getFifoTypeStore() const;
};

/// @brief Construct a DummySchedulerExtContext. In general prefer
/// SchedulerExtContext::dummyContext.
struct DummySchedulerExtContext : SchedulerExtContext {
  DummySchedulerExtContext() : SchedulerExtContext() {}
  bool isDummy() const override { return true; }
};

}  // namespace scheduler

#endif  // SCHEDULER_EXT_CONTEXT_H