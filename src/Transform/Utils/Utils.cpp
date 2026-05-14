//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
//===----------------------------------------------------------------------===//
//
// Utility functions for transform passes.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include <cassert>

#include "Ktdp/KtdpOps.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {
namespace utils {

bool isTargetConstant(int target, mlir::Value val) {
  if (auto const_op = val.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return const_op.value() == target;
  }
  return false;
}

mlir::Attribute convertKtdpToKtdfMemorySpace(
    mlir::OpBuilder& builder, mlir::Attribute ktdp_memory_space) {
  assert(ktdp_memory_space && "ktdp_memory_space attribute must not be null");
  auto spyre_attr =
      mlir::dyn_cast<mlir::ktdp::SpyreMemorySpaceAttr>(ktdp_memory_space);
  assert(spyre_attr && "ktdp_memory_space must be a SpyreMemorySpaceAttr");

  mlir::ktdp::SpyreMemorySpaceKind spyre_kind = spyre_attr.getValue();
  mlir::ktdf::MemorySpace ktdf_mem_space;
  switch (spyre_kind) {
    case mlir::ktdp::SpyreMemorySpaceKind::HBM:
      ktdf_mem_space = mlir::ktdf::MemorySpace::HBM;
      break;
    case mlir::ktdp::SpyreMemorySpaceKind::LX:
      ktdf_mem_space = mlir::ktdf::MemorySpace::LX;
      break;
    default:
      return nullptr;
  }

  return mlir::ktdf::MemorySpaceCustomAttr::get(builder.getContext(),
                                                ktdf_mem_space);
}

}  // namespace utils
}  // namespace scheduler

// Made with Bob
