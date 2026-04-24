#include <memory>
#include <set>

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"

#define PASS_NAME "compute-group-extraction"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace ktdf {
#define GEN_PASS_DEF_COMPUTEGROUPEXTRACTIONPASS
#include "src/Transform/Passes.h.inc"
}  // namespace ktdf

namespace {
const char VerboseDebug[] = DEBUG_TYPE "-verbose";

static llvm::cl::opt<bool> DisableThisPass(
    "compute-group-extraction-disable",
    llvm::cl::desc("Disable Compute Group Extraction pass"),
    llvm::cl::init(false));

}  // unnamed namespace

namespace {
struct ComputeGroupExtractionPass
    : public ::ktdf::impl::ComputeGroupExtractionPassBase<
          ComputeGroupExtractionPass> {
  void runOnOperation() final;

 private:
  void runOn(ModuleOp module_op);
  void collectComputeGroup(Operation* load_op,
                          std::set<Operation*>& compute_group);
  void collectOperands(const std::set<Operation*>& compute_group,
                      std::set<Operation*>& operands,
                      Block* for_body);
};

void ComputeGroupExtractionPass::collectComputeGroup(
    Operation* load_op, std::set<Operation*>& compute_group) {
  // Add the load operation itself
  compute_group.insert(load_op);

  // Traverse all uses of the load operation
  for (auto result : load_op->getResults()) {
    for (auto user : result.getUsers()) {
      if (compute_group.find(user) == compute_group.end()) {
        compute_group.insert(user);
        // Recursively collect users
        collectComputeGroup(user, compute_group);
      }
    }
  }
}

void ComputeGroupExtractionPass::collectOperands(
    const std::set<Operation*>& compute_group,
    std::set<Operation*>& operands, Block* for_body) {
  for (auto* op : compute_group) {
    for (auto operand : op->getOperands()) {
      auto* def_op = operand.getDefiningOp();
      if (def_op && compute_group.find(def_op) == compute_group.end()) {
        // Check if this operand is defined outside the for loop
        if (!for_body->findAncestorOpInBlock(*def_op)) {
          operands.insert(def_op);
          // Recursively collect operands of this operation
          collectOperands({def_op}, operands, for_body);
        }
      }
    }
  }
}

void ComputeGroupExtractionPass::runOn(ModuleOp module_op) {
  OpBuilder builder(module_op.getContext());

  int local_schedule_counter = 1;

  // Walk through all functions in the module
  module_op.walk([&](scf::ForOp for_op) {
      std::set<Operation*> compute_group;
      std::set<Operation*> operands;

      // Find all load operations in the for loop body
      for_op.getBody()->walk([&](Operation* op) {
        if (op->getName().getStringRef() == "ktdp.load") {
          collectComputeGroup(op, compute_group);
        }
      });

      if (compute_group.empty()) {
        return;
      }

      // Collect all operands needed by the compute group
      collectOperands(compute_group, operands, for_op.getBody());

      // Create a new module for the local schedule
      builder.setInsertionPointAfter(module_op);
      auto local_module = ModuleOp::create(module_op.getLoc());

      // Create the local-schedule function
      builder.setInsertionPointToStart(local_module.getBody());

      // Build function type with loop induction variable as argument
      llvm::SmallVector<Type> arg_types;
      arg_types.push_back(for_op.getInductionVar().getType());

      auto func_type = builder.getFunctionType(arg_types, {});
      auto local_func = func::FuncOp::create(
          for_op.getLoc(),
          "local-schedule-" + std::to_string(local_schedule_counter++),
          func_type);

      auto* entry_block = local_func.addEntryBlock();
      builder.setInsertionPointToStart(entry_block);

      // Create IRMapping for cloning operations
      IRMapping mapping;

      // Map the loop induction variable to the function argument
      mapping.map(for_op.getInductionVar(), local_func.getArgument(0));

      // Clone all operand operations first
      for (auto* op : operands) {
        auto* cloned_op = builder.clone(*op, mapping);
        // Update mapping for results
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
          mapping.map(op->getResult(i), cloned_op->getResult(i));
        }
      }

      // Clone all compute group operations
      for (auto* op : compute_group) {
        builder.clone(*op, mapping);
      }

      // Add return operation
      func::ReturnOp::create(builder, for_op.getLoc());

      LLVM_DEBUG(llvm::dbgs() << "Created local schedule function: "
                              << local_func.getName() << "\n");

      // Remove operations from the original for loop if they have no more uses
      for (auto* op : compute_group) {
        if (op->use_empty()) {
          op->erase();
        }
      }
  });
}

void ComputeGroupExtractionPass::runOnOperation() {
  if (DisableThisPass) return;
  LLVM_DEBUG(llvm::dbgs() << "Compute Group Extraction Pass\n");
  ModuleOp module_op = getOperation();
  runOn(module_op);
}

}  // namespace

namespace ktdf {

std::unique_ptr<mlir::Pass> createComputeGroupExtractionPass() {
  return std::make_unique<ComputeGroupExtractionPass>();
}

}  // namespace ktdf

