//===----------------------------- OpTraits.hpp----------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================

#ifndef OPTRAITS_HPP
#define OPTRAITS_HPP

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class ScalarValue : public TraitBase<ConcreteType, ScalarValue> {};

}  // namespace OpTrait
}  // namespace mlir

#endif  // OPTRAITS_HPP
