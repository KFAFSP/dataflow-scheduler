//=== OpTraits.td --- Dialect Operation Traits Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================

#ifndef OPTRAITS_TD
#define OPTRAITS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif  // OP_BASE

def ScalarValue : NativeOpTrait<"ScalarValue">;

#endif  // OPTRAITS_TD
