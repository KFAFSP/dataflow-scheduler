//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
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

#ifndef KTDF_CAPI_H
#define KTDF_CAPI_H

#include <mlir-c/Support.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(KTDF, ktdf);

//===----------------------------------------------------------------------===//
// TokenType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirKTDFTypeIsATokenType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDFTokenTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirKTDFTokenTypeGet(MlirContext c);

//===----------------------------------------------------------------------===//
// FifoSlotType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirKTDFTypeIsAFifoSlotType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDFFifoSlotTypeGetTypeID();

MLIR_CAPI_EXPORTED MlirType mlirKTDFFifoSlotTypeGet(uint32_t fifoType,
                                                    int64_t numElements,
                                                    MlirType elementType);

MLIR_CAPI_EXPORTED uint32_t mlirKTDFFifoSlotTypeGetFifoType(MlirType t);

MLIR_CAPI_EXPORTED int64_t mlirKTDFFifoSlotTypeGetNumElements(MlirType t);

MLIR_CAPI_EXPORTED MlirType mlirKTDFFifoSlotTypeGetElementType(MlirType t);

//===----------------------------------------------------------------------===//
// ApplicableUnitsAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirKTDFAttributeIsApplicableUnitsAttr(MlirAttribute a);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDFApplicableUnitsAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute mlirKTDFApplicableUnitsAttrGet(
    MlirContext ctx, intptr_t nComponentTypes, uint32_t const* componentTypes);

MLIR_CAPI_EXPORTED intptr_t
mlirKTDFApplicableUnitsAttrGetNumComponents(MlirAttribute a);

MLIR_CAPI_EXPORTED uint32_t
mlirKTDFApplicableUnitsAttrGetComponent(MlirAttribute a, intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif  // KTDF_CAPI_H