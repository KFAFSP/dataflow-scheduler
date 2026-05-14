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
#include <string>

#include "mlir/IR/Attributes.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

struct SchedulerExtContext {
  SchedulerExtContext() {}

  /// @brief Get a default initialized context for scheduler.
  /// The dummy context is intended for use with passes and analyses that
  /// require default constructors.
  static const SchedulerExtContext& dummyContext();

  virtual bool isDummy() const { return false; }

  /// Returns the compute unit type (SFP for current architecture)
  mlir::ktdf::ComponentType getComputeUnit() const;

  /// Returns the vector length in bytes for a given compute unit
  /// @param unit The compute unit type
  /// @return Vector length in bytes
  uint64_t getVectorLength(mlir::ktdf::ComponentType unit) const;

  /// Returns the maximum FIFO slot size for a given compute unit
  /// @param unit The compute unit type
  /// @return Maximum FIFO slot size in elements
  uint64_t getMaxFifoSlotSize(mlir::ktdf::ComponentType unit) const;

  /// Returns the interface type to the compute unit
  /// @param unit The compute unit type
  /// @return Interface type ("FIFO" or "MEM")
  std::string getInterfaceMemToComputeUnit(
      mlir::ktdf::ComponentType unit) const;

  /// Returns the FIFO type for load operations
  /// @return FifoType for loads (e.g., HbmToSfp)
  mlir::ktdf::FifoType getFifoTypeLoad() const;

  /// Returns the FIFO type for store operations
  /// @return FifoType for stores (e.g., SfpToHbm)
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