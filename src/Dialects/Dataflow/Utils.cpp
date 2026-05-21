//===----------------------- Utils.cpp ------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2022, 2023
//
// =============================================================================
//
// General utility functions and classes for the dataflow dialect
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"

#include <llvm/ADT/StringRef.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

// #include "dsc/designSpaceConfig.h"
#include "src/dt_exception.hpp"

using namespace mlir;
using namespace mlir::dataflow;

namespace dialects {
namespace dataflow {
namespace utils {

int getCoreId(const mlir::dataflow::GetUnitOp op) {
  DT_CHECK(op && "expected valid get_unit op");
  int core = -1;
  if (!op) return core;
  auto core_attr = op->getAttrOfType<mlir::IntegerAttr>("core");
  if (core_attr) {
    core = core_attr.getInt();
  }
  return core;
}

int getCoreletId(const mlir::dataflow::GetUnitOp op) {
  int corelet = -1;
  if (!op) return corelet;
  auto corelet_attr = op->getAttrOfType<mlir::IntegerAttr>("corelet");
  if (corelet_attr) {
    corelet = corelet_attr.getInt();
  }
  return corelet;
}

llvm::StringRef getUnitType(mlir::dataflow::GetUnitOp op) {
  DT_CHECK_MSG(op, "expected valid get_unit op");
  // auto record =
  //     DesignSpaceConfig::stringToSenComponents.find(op.getType().lower());
  // return DesignSpaceConfig::senCompToGenericComp.at(record->second);
  return op.getType();
}

std::string getUniqueUnitString(mlir::dataflow::GetUnitOp op) {
  DT_CHECK_MSG(op, "expected valid get_unit op");
  std::string result = op.getType().str();
  int coreid = getCoreId(op);
  if (coreid >= 0) {
    result += "_core" + std::to_string(coreid);
  }
  int coreletid = getCoreletId(op);
  if (coreletid >= 0) {
    result += "_corelet" + std::to_string(coreletid);
  }
  return result;
}

bool equivalent(const mlir::dataflow::GetUnitOp op1,
                const mlir::dataflow::GetUnitOp op2) {
  return getCoreId(op1) == getCoreId(op2) &&
         getCoreletId(op1) == getCoreletId(op2) &&
         getUnitType(op1) == getUnitType(op2);
}

template <typename LoopTy>
int getIterArgIndex(LoopTy& loop, BlockArgument& iter_arg) {
  DT_CHECK(iter_arg.getOwner()->getParentOp() == loop);
  auto region_args = loop.getRegionIterArgs();
  int index = -1;
  for (int i = 0, e = region_args.size(); i < e; ++i) {
    if (region_args[i] == iter_arg) {
      index = i;
      break;
    }
  }
  return index;
}

template <typename LoopTy>
Value getIterArgInit(LoopTy& loop, BlockArgument& iter_arg) {
  DT_CHECK(iter_arg.getOwner()->getParentOp() == loop);
  int index = getIterArgIndex<LoopTy>(loop, iter_arg);
  DT_CHECK(index >= 0);
  return loop.getOperand(loop.getNumControlOperands() + index);
}

Value getIterArgChainInit(BlockArgument& iter_arg, bool req_one_use) {
  Value val = iter_arg;
  while (auto arg = dyn_cast<BlockArgument>(val)) {
    DT_CHECK(!req_one_use || val.hasOneUse());
    auto parent_op = arg.getOwner()->getParentOp();
    if (auto affine_for = dyn_cast<affine::AffineForOp>(parent_op))
      val = getIterArgInit<affine::AffineForOp>(affine_for, arg);
    else if (auto scf_for = dyn_cast<scf::ForOp>(parent_op))
      val = getIterArgInit<scf::ForOp>(scf_for, arg);
    else
      llvm_unreachable("unsupported loop type");
  }
  return val;
}

void setIterArgChainInit(BlockArgument& iter_arg, Value replace_val) {
  Value val = iter_arg;
  while (auto arg = dyn_cast<BlockArgument>(val)) {
    auto parent_op = arg.getOwner()->getParentOp();
    if (auto affine_for = dyn_cast<affine::AffineForOp>(parent_op)) {
      val = getIterArgInit<affine::AffineForOp>(affine_for, arg);
      if (!isa<BlockArgument>(val)) {
        auto idx = getIterArgIndex<affine::AffineForOp>(affine_for, arg);
        affine_for->setOperand(idx, replace_val);
      }
    } else if (auto scf_for = dyn_cast<scf::ForOp>(parent_op)) {
      val = getIterArgInit<scf::ForOp>(scf_for, arg);
      if (!isa<BlockArgument>(val)) {
        auto idx = getIterArgIndex<scf::ForOp>(scf_for, arg);
        scf_for->setOperand(idx, replace_val);
      }
    } else {
      llvm_unreachable("unsupported loop type");
    }
  }
}

void setIterArgInit(BlockArgument& iter_arg, Value replace_val) {
  int operand_idx = -1;
  auto parent_op = iter_arg.getOwner()->getParentOp();
  if (auto affine_for = dyn_cast<affine::AffineForOp>(parent_op)) {
    auto operand_idx =
        getIterArgIndex<affine::AffineForOp>(affine_for, iter_arg);
    affine_for.setOperand(affine_for.getNumControlOperands() + operand_idx,
                          replace_val);
  } else {
    auto scf_for = dyn_cast<scf::ForOp>(parent_op);
    DT_CHECK_MSG(scf_for, "unexpected owning operation");
    auto operand_idx = getIterArgIndex<scf::ForOp>(scf_for, iter_arg);
    scf_for.setOperand(scf_for.getNumControlOperands() + operand_idx,
                       replace_val);
  }
}

SmallVector<BlockArgument> getIterArgChain(BlockArgument& iter_arg,
                                           Value& init_val) {
  SmallVector<BlockArgument> iter_arg_chain;
  BlockArgument curr_arg = iter_arg;
  while (curr_arg) {
    iter_arg_chain.push_back(curr_arg);

    auto* loop = curr_arg.getOwner()->getParentOp();
    if (auto affine_for = dyn_cast<affine::AffineForOp>(loop)) {
      init_val = getIterArgInit<affine::AffineForOp>(affine_for, curr_arg);
    } else {
      auto scf_for = dyn_cast<scf::ForOp>(loop);
      DT_CHECK(scf_for);
      init_val = getIterArgInit<scf::ForOp>(scf_for, curr_arg);
    }
    curr_arg = dyn_cast_or_null<BlockArgument>(init_val);
  }
  return iter_arg_chain;
}

}  // namespace utils
}  // namespace dataflow
}  // namespace dialects
