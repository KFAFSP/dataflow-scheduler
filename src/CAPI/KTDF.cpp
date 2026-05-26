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

#include "src/CAPI/KTDF.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <mlir/CAPI/IR.h>

#include "mlir/CAPI/Registration.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Dialects/KTDF/KTDFTypes.hpp"

using namespace mlir;
using namespace mlir::ktdf;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(KTDF, ktdf, mlir::ktdf::KTDFDialect)

//===----------------------------------------------------------------------===//
// TokenType
//===----------------------------------------------------------------------===//

bool mlirKTDFTypeIsATokenType(MlirType t) { return isa<TokenType>(unwrap(t)); }

MlirTypeID mlirKTDFTokenTypeGetTypeID(void) {
  return wrap(TokenType::getTypeID());
}

MlirType mlirKTDFTokenTypeGet(MlirContext c) {
  return wrap(TokenType::get(unwrap(c)));
}

//===----------------------------------------------------------------------===//
// FifoSlotType
//===----------------------------------------------------------------------===//

bool mlirKTDFTypeIsAFifoSlotType(MlirType t) {
  return isa<FifoSlotType>(unwrap(t));
}

MlirTypeID mlirKTDFFifoSlotTypeGetTypeID(void) {
  return wrap(FifoSlotType::getTypeID());
}

MlirType mlirKTDFFifoSlotTypeGet(uint32_t fifoType, int64_t numElements,
                                 MlirType elementType) {
  return wrap(FifoSlotType::get(unwrap(elementType).getContext(),
                                static_cast<FifoType>(fifoType), numElements,
                                unwrap(elementType)));
}

MLIR_CAPI_EXPORTED uint32_t mlirKTDFFifoSlotTypeGetFifoType(MlirType t) {
  return static_cast<uint32_t>(cast<FifoSlotType>(unwrap(t)).getFifoType());
}

MLIR_CAPI_EXPORTED int64_t mlirKTDFFifoSlotTypeGetNumElements(MlirType t) {
  return cast<FifoSlotType>(unwrap(t)).getNumElements();
}

MLIR_CAPI_EXPORTED MlirType mlirKTDFFifoSlotTypeGetElementType(MlirType t) {
  return wrap(cast<FifoSlotType>(unwrap(t)).getElementType());
}

//===----------------------------------------------------------------------===//
// ApplicableUnitsAttr
//===----------------------------------------------------------------------===//

bool mlirKTDFAttributeIsApplicableUnitsAttr(MlirAttribute a) {
  return isa<ApplicableUnitsAttr>(unwrap(a));
}

MlirTypeID mlirKTDFApplicableUnitsAttrGetTypeID(void) {
  return wrap(ApplicableUnitsAttr::getTypeID());
}

MlirAttribute mlirKTDFApplicableUnitsAttrGet(MlirContext ctx,
                                             intptr_t nComponentTypes,
                                             uint32_t const* componentTypes) {
  return wrap(ApplicableUnitsAttr::get(
      unwrap(ctx), llvm::to_vector(llvm::map_range(
                       ArrayRef(componentTypes, nComponentTypes),
                       [](auto x) { return static_cast<ComponentType>(x); }))));
}

intptr_t mlirKTDFApplicableUnitsAttrGetNumComponents(MlirAttribute a) {
  return cast<ApplicableUnitsAttr>(unwrap(a)).getValue().size();
}

uint32_t mlirKTDFApplicableUnitsAttrGetComponent(MlirAttribute a,
                                                 intptr_t pos) {
  return static_cast<uint32_t>(
      cast<ApplicableUnitsAttr>(unwrap(a)).getValue()[pos]);
}
