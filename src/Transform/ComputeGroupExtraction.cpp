//===------------------------- ComputeGroupExtraction.cpp ----------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2022, 2025
//
// =============================================================================
//
/// This pass identifies compute groups (sequences of operations between LoadOps
/// and StoreOps with the same access tile equivalence class) and extracts them
/// into separate functions in new modules.
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpOps.hpp"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  /// A compute group is defined as the set of operations between
  /// first_load and last_store, in addition to any ancestor operations.
  struct ComputeGroup {
    Operation* first_load;
    Operation* last_store;
  };

  void runOn(ModuleOp module_op);

  /// Walks the ancestor tree of a StoreOp's data operand and unions access
  /// tiles from any LoadOps found with the StoreOp's access tile.
  /// @param store_op The StoreOp whose ancestors to walk
  /// @param access_tile_eq_classes Equivalence classes to update
  void unionAccessTilesFromAncestors(
      mlir::ktdp::StoreOp store_op,
      llvm::EquivalenceClasses<Value>& access_tile_eq_classes);

  /// Print equivalence classes for debugging.
  /// @param access_tile_eq_classes The equivalence classes to print
  void printEqClasses(llvm::EquivalenceClasses<Value>& access_tile_eq_classes);

  /// Verifiy that a compute operation has valid usage patterns.
  /// @param op The operation to verify
  /// @return true if the operation has valid uses, false otherwise
  bool verifyComputeOpUses(Operation* op);

  /// Build equivalence classes of access tiles by processing StoreOps in a
  /// block.
  /// @param block The block to process
  /// @param access_tile_eq_classes Equivalence classes to populate
  void buildAccessTileEqClasses(
      Block* block, llvm::EquivalenceClasses<Value>& access_tile_eq_classes);

  /// Extract a compute group into a new function in a child module.
  /// @param top_level_module The top-level module containing all child modules
  /// @param original_module The module containing the original function
  /// @param group The compute group to extract
  void extractComputeGroup(ModuleOp top_level_module, ModuleOp original_module,
                           const ComputeGroup& group);

  /// Process a block to identify and collect compute groups for extraction.
  /// @param block The block to process
  /// @param access_tile_eq_classes Rquivalence classes of access tiles
  void processBlockForExtraction(
      Block* block, llvm::EquivalenceClasses<Value>& access_tile_eq_classes);

  /// Counter for generating unique function names
  int function_counter = 0;
  /// Collected compute groups
  llvm::SmallVector<ComputeGroup> groups_to_extract_;
};

void ComputeGroupExtractionPass::unionAccessTilesFromAncestors(
    mlir::ktdp::StoreOp store_op,
    llvm::EquivalenceClasses<Value>& access_tile_eq_classes) {
  Value store_access_tile = store_op.getAccessTile();
  Block* store_block = store_op->getBlock();

  llvm::SmallVector<Operation*> worklist = {store_op};
  llvm::DenseSet<Operation*> visited;

  // Starting at the store, walk all the ancestors, searching for loads
  // and union the store's access tile equivalence class with the class
  // of the load's access tile. This captures the dependencies between
  // loads and stores.
  while (!worklist.empty()) {
    Operation* current_op = worklist.back();
    worklist.pop_back();

    if (visited.count(current_op)) continue;
    visited.insert(current_op);

    if (auto load_op = dyn_cast<mlir::ktdp::LoadOp>(current_op)) {
      Value load_access_tile = load_op.getAccessTile();
      assert(load_op->getBlock() == store_block &&
             "LoadOp expected to be in same block as StoreOp");
      access_tile_eq_classes.unionSets(store_access_tile, load_access_tile);
      continue;
    }

    for (Value operand : current_op->getOperands()) {
      if (Operation* operand_def_op = operand.getDefiningOp()) {
        // Special handling for tensor.empty() - follow its uses to find compute
        // op.
        if (isa<tensor::EmptyOp>(operand_def_op)) {
          // Push all uses of tensor.empty() except stores and current_op
          for (Operation* user : operand_def_op->getUsers()) {
            if (user == current_op || isa<mlir::ktdp::StoreOp>(user)) {
              continue;
            }
            worklist.push_back(user);
          }
          continue;
        }
        worklist.push_back(operand_def_op);
      }
    }
  }
}

bool ComputeGroupExtractionPass::verifyComputeOpUses(Operation* op) {
  bool has_store_use = false;
  bool has_compute_use = false;

  for (Value result : op->getResults()) {
    for (Operation* user : result.getUsers()) {
      if (isa<mlir::ktdp::StoreOp>(user)) {
        has_store_use = true;
      } else if (llvm::isa<mlir::linalg::LinalgDialect>(user->getDialect()) ||
                 llvm::isa<mlir::math::MathDialect>(user->getDialect()) ||
                 llvm::isa<mlir::arith::ArithDialect>(user->getDialect())) {
        has_compute_use = true;
      }
    }
  }

  if (has_store_use && has_compute_use) return false;

  return true;
}

void ComputeGroupExtractionPass::printEqClasses(
    llvm::EquivalenceClasses<Value>& access_tile_eq_classes) {
  for (llvm::EquivalenceClasses<Value>::iterator
           I = access_tile_eq_classes.begin(),
           E = access_tile_eq_classes.end();
       I != E; ++I) {
    if (!(*I)->isLeader()) continue;

    llvm::dbgs() << "  Class leader: " << (*I)->getData() << "\n";
    llvm::dbgs() << "    Members:\n";

    for (llvm::EquivalenceClasses<Value>::member_iterator MI =
             access_tile_eq_classes.member_begin(**I);
         MI != access_tile_eq_classes.member_end(); ++MI) {
      llvm::dbgs() << "    " << *MI << "\n";
    }
    llvm::dbgs() << "\n-------- END OF CLASS MEMBERS -------\n";
  }
}

void ComputeGroupExtractionPass::buildAccessTileEqClasses(
    Block* block, llvm::EquivalenceClasses<Value>& access_tile_eq_classes) {
  for (Operation& block_op : block->getOperations()) {
    if (auto store_op = dyn_cast<mlir::ktdp::StoreOp>(&block_op)) {
      // For every store in block, union its access tile's class with the
      // classes of the access tiles of its ancestor loads.
      access_tile_eq_classes.insert(store_op.getAccessTile());
      unionAccessTilesFromAncestors(store_op, access_tile_eq_classes);
    } else if (llvm::isa<mlir::linalg::LinalgDialect>(block_op.getDialect()) ||
               llvm::isa<mlir::math::MathDialect>(block_op.getDialect()) ||
               llvm::isa<mlir::arith::ArithDialect>(block_op.getDialect())) {
      assert(verifyComputeOpUses(&block_op) &&
             "Compute operation must have valid usage pattern");
    }
  }
}

// Helper function to recursively materialize dependencies
static void materializeDependency(Value val, IRMapping& mapper,
                                  OpBuilder& builder) {
  // If we already cloned this ancestor, we're done
  if (mapper.contains(val)) return;
  assert(!isa<BlockArgument>(val) && "Not expecting to clone iter args");

  Operation* op = val.getDefiningOp();

  // Ensure all operands of the ancestor are cloned first
  for (Value operand : op->getOperands()) {
    materializeDependency(operand, mapper, builder);
  }

  // Now clone the ancestor itself into the new block
  builder.clone(*op, mapper);
}

void ComputeGroupExtractionPass::extractComputeGroup(
    ModuleOp top_level_module, ModuleOp original_module,
    const ComputeGroup& group) {
  assert(group.first_load && group.last_store);

  std::string func_name =
      "local-schedule-" + std::to_string(function_counter++);

  OpBuilder builder(top_level_module);

  // Create forward declaration in the original module
  builder.setInsertionPointToEnd(original_module.getBody());
  auto func_type = builder.getFunctionType({}, {});
  auto forward_decl =
      func::FuncOp::create(top_level_module.getLoc(), func_name, func_type);
  forward_decl.setPrivate();
  builder.insert(forward_decl);

  // Create a new child module in the top-level module for this extracted
  // function
  builder.setInsertionPointToEnd(top_level_module.getBody());
  auto extracted_module = ModuleOp::create(builder, top_level_module.getLoc());

  // Create the function definition in the extracted module
  builder.setInsertionPointToStart(extracted_module.getBody());
  auto func_op =
      func::FuncOp::create(top_level_module.getLoc(), func_name, func_type);
  func_op.setPrivate();
  extracted_module.push_back(func_op);

  // Create entry block
  Block* func_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(func_block);

  // Collect operations to move (from first_load to last_store inclusive)
  llvm::SmallVector<Operation*> ops_to_move;
  bool collecting = false;
  Block* group_block = group.first_load->getBlock();
  for (Operation& op : group_block->getOperations()) {
    if (&op == group.first_load) {
      collecting = true;
    }
    if (collecting) {
      ops_to_move.push_back(&op);
    }
    if (&op == group.last_store) {
      break;
    }
  }

  // Create call operation before moving operations
  builder.setInsertionPoint(group.first_load);
  auto call_op = func::CallOp::create(builder, top_level_module.getLoc(),
                                      forward_decl, {});

  builder.setInsertionPointToStart(func_block);
  IRMapping mapper;

  // Step 1: Materialize all dependencies for the entire range.
  // (e.g. arith.constants, construct_memory_views, construct_access_tiles)
  for (Operation* op : ops_to_move) {
    for (Value operand : op->getOperands()) {
      materializeDependency(operand, mapper, builder);
    }
  }

  // Step 2: Clone the actual range
  for (Operation* op : ops_to_move) {
    // Skip if already cloned as dependency
    bool already_cloned = false;
    for (Value result : op->getResults()) {
      if (mapper.contains(result)) {
        already_cloned = true;
        break;
      }
    }

    if (!already_cloned) {
      builder.clone(*op, mapper);
    }
  }

  // Step 3: Erase the original operations in reverse order to avoid dangling
  // uses
  for (auto it = ops_to_move.rbegin(); it != ops_to_move.rend(); ++it) {
    Operation* op = *it;
    if (op->use_empty()) {
      op->erase();
    }
  }

  // Add return operation to the new function
  builder.setInsertionPointToEnd(func_block);
  func::ReturnOp::create(builder, top_level_module.getLoc());

  LLVM_DEBUG({
    llvm::dbgs() << "Extracted compute group into function: " << func_name
                 << "\n";
    llvm::dbgs() << "Operations: " << ops_to_move.size() << "\n";
  });
}

void ComputeGroupExtractionPass::processBlockForExtraction(
    Block* block, llvm::EquivalenceClasses<Value>& access_tile_eq_classes) {
  // Keep track of visited equivalence classes of access tiles for the current
  // block to assert they are not visited multiple times.
  llvm::DenseSet<Value> visited_class_leaders;
  Value current_class_leader = nullptr;
  bool last_mem_op_was_load = false;
  Operation* first_load_op = nullptr;
  Operation* last_store_op = nullptr;

  // Traverse operations in the block, extracting each compute group. Boundaries
  // are determined by changes in current equivalence class of load/store access
  // tiles and/or control flow operations.
  for (Operation& block_op : block->getOperations()) {
    // Check for control flow operations - they terminate the current compute
    // group.
    // Reason: Otherwise, if control flow operations were to be extracted,
    // we might end up with nested func calls.
    if (isa<scf::ForOp>(&block_op) || isa<scf::IfOp>(&block_op)) {
      // Save current compute group if any
      if (current_class_leader && first_load_op && last_store_op) {
        groups_to_extract_.push_back({first_load_op, last_store_op});

        // Mark current class as visited
        visited_class_leaders.insert(current_class_leader);

        // Reset tracking
        current_class_leader = nullptr;
        first_load_op = nullptr;
        last_store_op = nullptr;
        last_mem_op_was_load = false;
      }
      continue;
    }

    if (auto load_op = dyn_cast<mlir::ktdp::LoadOp>(&block_op)) {
      Value load_access_tile = load_op.getAccessTile();
      assert(access_tile_eq_classes.findLeader(load_access_tile) !=
                 access_tile_eq_classes.member_end() &&
             "expecting access_tile to have equivalence class");
      Value load_class_leader =
          access_tile_eq_classes.getLeaderValue(load_access_tile);
      assert(visited_class_leaders.find(load_class_leader) ==
                 visited_class_leaders.end() &&
             "Equivalence class visited twice");

      // If this is a different class than current, save previous group and
      // switch
      if (current_class_leader && current_class_leader != load_class_leader) {
        assert(!last_mem_op_was_load &&
               "found consecutive loads in different equivalence classes");

        // Save the previous compute group
        if (first_load_op && last_store_op) {
          groups_to_extract_.push_back({first_load_op, last_store_op});
        }

        // Mark previous class as visited before switching
        visited_class_leaders.insert(current_class_leader);

        current_class_leader = load_class_leader;
        first_load_op = nullptr;
        last_store_op = nullptr;
      } else if (!current_class_leader) {
        // First load - initialize current class
        current_class_leader = load_class_leader;
      }

      // Track first load of current class
      if (!first_load_op) {
        first_load_op = &block_op;
      }

      last_mem_op_was_load = true;
    } else if (auto store_op = dyn_cast<mlir::ktdp::StoreOp>(&block_op)) {
      Value store_access_tile = store_op.getAccessTile();

      assert(access_tile_eq_classes.findLeader(store_access_tile) !=
                 access_tile_eq_classes.member_end() &&
             "expecting access_tile to have equivalence class");

      Value store_class_leader =
          access_tile_eq_classes.getLeaderValue(store_access_tile);

      assert(current_class_leader && "StoreOp found before any LoadOp");

      assert(current_class_leader == store_class_leader &&
             "StoreOp must belong to the same class as previous LoadOp");

      // Track last store of current class
      last_store_op = &block_op;
      last_mem_op_was_load = false;
    }
  }

  // Save the last compute group
  if (first_load_op && last_store_op) {
    groups_to_extract_.push_back({first_load_op, last_store_op});
  }
}

void ComputeGroupExtractionPass::runOn(ModuleOp module_op) {
  // Walk all blocks and collect compute groups to extract
  module_op.walk([&](Block* block) {
    llvm::EquivalenceClasses<Value> access_tile_eq_classes;

    buildAccessTileEqClasses(block, access_tile_eq_classes);

    if (access_tile_eq_classes.empty()) {
      return WalkResult::advance();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Access tile equivalence classes:\n";
      printEqClasses(access_tile_eq_classes);
    });

    // Process block to collect compute groups
    processBlockForExtraction(block, access_tile_eq_classes);

    return WalkResult::advance();
  });

  // If no groups to extract, nothing to do
  if (groups_to_extract_.empty()) {
    return;
  }

  // Move original content to a child module and create child
  // modules for extracted functions. New structure:
  // module {
  //   module { // original
  //     func.func @orig {
  //       <original ops with extracted code replaced by func call>
  //     }
  //   }
  //   module { // extracted module #1
  //     ...
  //   }
  //   ... // other extracted modules
  // }
  OpBuilder builder(module_op);

  // Create a child module for the original content
  builder.setInsertionPointToStart(module_op.getBody());
  auto original_module = ModuleOp::create(builder, module_op.getLoc());

  // Move all operations from the original module to the new child module
  auto& original_ops = module_op.getBody()->getOperations();
  auto& child_ops = original_module.getBody()->getOperations();
  child_ops.splice(child_ops.begin(), original_ops,
                   std::next(original_module->getIterator()),
                   original_ops.end());

  // Now extract all collected compute groups, each into its own child module
  for (const auto& group : groups_to_extract_) {
    extractComputeGroup(module_op, original_module, group);
  }
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
