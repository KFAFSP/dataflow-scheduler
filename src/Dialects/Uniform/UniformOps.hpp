//===--------------------------- UniformOps.hpp----------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2023
//
// =============================================================================
//
// This file declares the Uniform operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_DIALECT_UNIFORM_HPP_
#define DIALECTS_DIALECT_UNIFORM_HPP_

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

/// Auto-generated includes.
#include "src/Dialects/Uniform/UniformOpsDialect.hpp.inc"

namespace mlir::uniform {

static StringRef getRegLocalesAttrStrName() { return "regLocales"; }
static StringRef getRegIndicesAttrStrName() { return "regIndices"; }

}  // namespace mlir::uniform

#define GET_OP_CLASSES
#include "src/Dialects/Uniform/UniformOps.hpp.inc"

#endif  // DIALECTS_DIALECT_UNIFORM_HPP_
