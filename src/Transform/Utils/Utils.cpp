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
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
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

  return mlir::ktdf::MemorySpaceAttr::get(builder.getContext(), ktdf_mem_space);
}

std::optional<int64_t> getStaticTripCount(mlir::scf::ForOp loop) {
  auto lb_const =
      loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto ub_const =
      loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto step_const =
      loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!lb_const || !ub_const || !step_const) {
    return std::nullopt;
  }
  int64_t lb = lb_const.value();
  int64_t ub = ub_const.value();
  int64_t step = step_const.value();
  if (step <= 0) {
    return std::nullopt;
  }
  if (ub <= lb) {
    return 0;
  }
  return (ub - lb + step - 1) / step;
}

bool isParallelLoop(mlir::scf::ForOp loop) {
  auto attr = loop->getAttrOfType<mlir::ktdf::LoopTypeAttr>("loop_type");
  if (!attr) {
    return false;
  }
  return attr.getValue() == mlir::ktdf::LoopType::ParallelLoop;
}

mlir::LogicalResult cleanupPrivateOp(mlir::ktdf::PrivateOp private_op) {
  mlir::Block* body = private_op.getBodyBlock();
  auto yield_op = private_op.getYieldOp();

  llvm::SmallVector<unsigned> live_result_indices;
  llvm::SmallVector<mlir::Type> live_result_types;
  llvm::SmallVector<mlir::Value> live_yield_operands;

  for (auto [idx, result] : llvm::enumerate(private_op.getResults())) {
    if (result.use_empty()) continue;
    live_result_indices.push_back(idx);
    live_result_types.push_back(result.getType());
    live_yield_operands.push_back(yield_op.getOperands()[idx]);
  }

  if (live_result_indices.size() == private_op.getNumResults()) {
    return mlir::success();
  }

  llvm::SetVector<mlir::Operation*> needed_ops;
  llvm::SmallVector<mlir::Value> worklist(live_yield_operands.begin(),
                                          live_yield_operands.end());

  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto op_result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!op_result) continue;

    mlir::Operation* def_op = op_result.getOwner();
    if (def_op->getBlock() != body || def_op == yield_op) continue;
    if (!needed_ops.insert(def_op)) continue;

    for (mlir::Value operand : def_op->getOperands()) {
      worklist.push_back(operand);
    }
  }

  mlir::OpBuilder builder(private_op);
  auto new_private = mlir::ktdf::PrivateOp::create(builder, private_op.getLoc(),
                                                   live_result_types);
  mlir::Block& new_body = new_private.getBody().front();

  mlir::IRMapping mapping;
  builder.setInsertionPointToStart(&new_body);
  for (mlir::Operation& op : body->without_terminator()) {
    if (!needed_ops.count(&op)) continue;
    builder.clone(op, mapping);
  }

  llvm::SmallVector<mlir::Value> new_yield_operands;
  new_yield_operands.reserve(live_yield_operands.size());
  for (mlir::Value operand : live_yield_operands) {
    new_yield_operands.push_back(mapping.lookupOrDefault(operand));
  }
  mlir::ktdf::PrivateYieldOp::create(builder, yield_op.getLoc(),
                                     new_yield_operands);

  for (auto [new_idx, old_idx] : llvm::enumerate(live_result_indices)) {
    private_op.getResult(old_idx).replaceAllUsesWith(
        new_private.getResult(new_idx));
  }

  private_op.erase();
  return mlir::success();
}

mlir::LogicalResult cleanupPrivateOpsInPipeline(
    mlir::ktdf::PipelineOp pipeline_op) {
  llvm::SmallVector<mlir::ktdf::PrivateOp> private_ops;
  for (mlir::Operation& op : pipeline_op.getBodyBlock()->getOperations()) {
    if (auto private_op = mlir::dyn_cast<mlir::ktdf::PrivateOp>(&op)) {
      private_ops.push_back(private_op);
    }
  }

  for (mlir::ktdf::PrivateOp private_op : private_ops) {
    if (mlir::failed(cleanupPrivateOp(private_op))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

}  // namespace utils
}  // namespace scheduler

// Made with Bob
