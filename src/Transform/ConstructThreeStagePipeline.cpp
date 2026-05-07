//===------------------- ConstructThreeStagePipeline.cpp -----------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
// This pass constructs a three-stage pipeline.
//
// First it transforms compute operations on tensors into nested scf.for loops
// operating on memref buffers. Then it constructs a three-stage pipeline (load
// stage, compute stage, store stage).
//
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpOps.hpp"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"
#include "src/Utils/SchedulerExtContext.h"

#define PASS_NAME "construct-three-stage-pipeline"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace ktdf {
#define GEN_PASS_DEF_CONSTRUCTTHREESTAGEPIPELINEPASS
#include "src/Transform/Passes.h.inc"
}  // namespace ktdf

namespace {

static llvm::cl::opt<bool> DisableThisPass(
    "construct-three-stage-pipeline-disable",
    llvm::cl::desc("Disable construction of three stage pipeline"),
    llvm::cl::init(false));

}  // unnamed namespace

namespace {

struct ConstructThreeStagePipelinePass
    : public ::ktdf::impl::ConstructThreeStagePipelinePassBase<
          ConstructThreeStagePipelinePass> {
  ConstructThreeStagePipelinePass(
      const scheduler::SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    ConstructThreeStagePipelinePassBase::getDependentDialects(registry);
    // Register bufferization interface implementations
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() final;

 private:
  const scheduler::SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }
  const scheduler::SchedulerExtContext& scheduler_ctx_;

  // Collected ktdp.load and ktdp.store operations
  SmallVector<ktdp::LoadOp> load_ops_;
  SmallVector<ktdp::StoreOp> store_ops_;
  SmallVector<linalg::LinalgOp> compute_ops_;

  // Tiled loops from linalg tiling (outermost to innermost)
  SmallVector<Operation*> tiled_loops_;

  // Operations to delete after pipeline creation
  SmallVector<Operation*> ops_to_delete_;

  // Builder for constants at function start
  std::optional<OpBuilder> const_builder_;

  // Generalize named linalg operations to linalg.generic
  LogicalResult generalizeLinalgOps(ModuleOp module);

  // Fuse consecutive linalg.generic operations
  LogicalResult fuseLinalgOps(ModuleOp module);

  // Determine tile sizes based on stick_size from linalg operation
  SmallVector<int64_t> determineTileSizes(linalg::LinalgOp linalgOp);

  // Compute memref shape from tile sizes (drop leading 1s)
  SmallVector<int64_t> determineMemrefShape(
      const SmallVector<int64_t>& tile_sizes);

  // Transform linalg operations to loops
  void lowerLinalgToLoops(ModuleOp module);

  // Create a 3-stage pipeline inside innermost_loop, with one stage for loads,
  // computes, stores.
  void createPipeline(scf::ForOp innermost_loop,
                      const SmallVector<int64_t>& tile_sizes);

  // Create linalg compute operations in stage 2
  void createComputeOps(OpBuilder& builder, Location loc,
                        mlir::ktdf::PrivateOp private_op);

  // Create data transfer operations for loads or stores
  // is_load=true: transfer from access_tile to private memref
  // is_load=false: transfer from private memref to access_tile
  // private_result_offset: index into private_op results where memrefs for this
  //   transfer start. For loads, offset=0 (load memrefs come first). For
  //   stores, offset=load_ops_.size() (store memrefs come after load memrefs).
  void createDataTransfers(OpBuilder& builder, Location loc,
                           mlir::ktdf::PrivateOp private_op,
                           ArrayRef<int64_t> memref_shape, bool is_load,
                           size_t private_result_offset);

  // Delete op and recursively delete unused ancestors
  void deleteOpAndUnusedAncestors(Operation* op);

  // Replace construct_access_tile with memref.reinterpret_cast
  void replaceAccessTilesWithReinterpretCast(ModuleOp module);

  // Clean up operations after pipeline creation
  void cleanupOperations();
};

LogicalResult ConstructThreeStagePipelinePass::generalizeLinalgOps(
    ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs()
             << "Generalizing named linalg operations to linalg.generic\n");

  SmallVector<linalg::LinalgOp> linalg_ops;
  module.walk([&](linalg::LinalgOp linalg_op) {
    // Skip if already generic
    if (!isa<linalg::GenericOp>(linalg_op.getOperation())) {
      linalg_ops.push_back(linalg_op);
    }
  });

  IRRewriter rewriter(&getContext());
  for (auto linalg_op : linalg_ops) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Generalizing: " << linalg_op->getName() << "\n");
    rewriter.setInsertionPoint(linalg_op);

    FailureOr<linalg::GenericOp> generic_result =
        linalg::generalizeNamedOp(rewriter, linalg_op);

    if (failed(generic_result)) {
      linalg_op.emitError("Failed to generalize linalg operation");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  Successfully generalized to linalg.generic\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "After generalization:\n" << module << "\n\n");
  return success();
}

LogicalResult ConstructThreeStagePipelinePass::fuseLinalgOps(ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Fusing consecutive linalg.generic operations\n");

  // Use a worklist to process operations
  SmallVector<linalg::GenericOp> worklist;
  module.walk(
      [&](linalg::GenericOp generic_op) { worklist.push_back(generic_op); });

  LLVM_DEBUG(llvm::dbgs() << "Found " << worklist.size()
                          << " linalg.generic operations\n");

  IRRewriter rewriter(&getContext());

  while (!worklist.empty()) {
    auto consumer = worklist.pop_back_val();

    // Try to fuse producer into this consumer
    for (auto* operand : consumer.getDpsInputOperands()) {
      Value input = operand->get();
      auto producer = input.getDefiningOp<linalg::GenericOp>();

      if (!producer) continue;

      rewriter.setInsertionPoint(consumer);
      FailureOr<linalg::ElementwiseOpFusionResult> fusion_result =
          linalg::fuseElementwiseOps(rewriter, operand);

      if (succeeded(fusion_result)) {
        LLVM_DEBUG(llvm::dbgs() << "  Successfully fused operations\n");

        // Get the fused operation
        auto fused_op = cast<linalg::GenericOp>(fusion_result->fusedOp);

        // Replace uses of consumer with the fused operation (this also erases
        // consumer)
        rewriter.replaceOp(consumer, fused_op->getResults());

        // Erase the original producer operation
        rewriter.eraseOp(producer);

        // Remove producer from worklist if present
        llvm::erase(worklist, producer);

        // Add the fused operation to the worklist for further fusion
        worklist.push_back(fused_op);

        break;  // Process the fused op in the next iteration
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "After fusion:\n" << module << "\n\n");
  return success();
}

SmallVector<int64_t> ConstructThreeStagePipelinePass::determineTileSizes(
    linalg::LinalgOp linalg_op) {
  const int64_t stick_size = schedulerExtContext().getStickSize();
  SmallVector<int64_t> tile_sizes;

  assert(linalg_op->getNumResults() > 0 &&
         "linalg op must have tensor results");

  ShapedType shaped_type =
      llvm::dyn_cast<ShapedType>(linalg_op->getResult(0).getType());
  assert(shaped_type && shaped_type.hasRank());
  ArrayRef<int64_t> shape = shaped_type.getShape();
  int64_t rank = shape.size();

  // Start from rightmost dimension and multiply until we reach stick_size
  int64_t product = 1;
  int64_t covered_dims = 0;

  for (int64_t i = rank - 1; i >= 0; --i) {
    product *= shape[i];
    covered_dims++;
    if (product >= stick_size) {
      break;
    }
  }

  // Build tile sizes: 1 for uncovered dims, partial/full for covered dims
  tile_sizes.resize(rank);
  for (int64_t i = 0; i < rank - covered_dims; ++i) {
    tile_sizes[i] = 1;
  }

  // For covered dimensions, compute tile sizes that divide evenly
  int64_t remaining_product = stick_size;
  for (int64_t i = rank - 1; i >= rank - covered_dims; --i) {
    int64_t dim_size = shape[i];

    // Use GCD to find largest value that divides both dim_size and
    // remaining_product
    int64_t tile_size = std::gcd(dim_size, remaining_product);

    // Clamp to dimension size
    tile_size = std::min(tile_size, dim_size);

    tile_sizes[i] = tile_size;
    remaining_product /= tile_size;
    if (remaining_product <= 1) remaining_product = 1;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    Shape: [";
    for (int64_t i = 0; i < rank; ++i) {
      llvm::dbgs() << shape[i];
      if (i < rank - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n    Tile sizes: [";
    for (int64_t i = 0; i < rank; ++i) {
      llvm::dbgs() << tile_sizes[i];
      if (i < rank - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
  });

  return tile_sizes;
}

SmallVector<int64_t> ConstructThreeStagePipelinePass::determineMemrefShape(
    const SmallVector<int64_t>& tile_sizes) {
  SmallVector<int64_t> memref_shape;
  for (int64_t tile_size : tile_sizes) {
    if (tile_size != 1 || !memref_shape.empty()) {
      memref_shape.push_back(tile_size);
    }
  }
  if (memref_shape.empty()) {
    memref_shape.push_back(1);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  Memref shape from tile sizes: [";
    for (size_t i = 0; i < memref_shape.size(); ++i) {
      llvm::dbgs() << memref_shape[i];
      if (i < memref_shape.size() - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
  });

  return memref_shape;
}

void ConstructThreeStagePipelinePass::lowerLinalgToLoops(ModuleOp module) {
  LLVM_DEBUG(
      llvm::dbgs() << "Lowering linalg.generic operations to SCF loops\n");

  // Collect ktdp load/store operations before any transformations
  module.walk([&](Operation* op) {
    if (auto load_op = dyn_cast<ktdp::LoadOp>(op)) {
      load_ops_.push_back(load_op);
      ops_to_delete_.push_back(op);
    } else if (auto store_op = dyn_cast<ktdp::StoreOp>(op)) {
      store_ops_.push_back(store_op);
      ops_to_delete_.push_back(op);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "  Found " << load_ops_.size()
                          << " ktdp.load operations\n");
  LLVM_DEBUG(llvm::dbgs() << "  Found " << store_ops_.size()
                          << " ktdp.store operations\n");

  // Collect linalg operations (should be linalg.generic after generalization)
  SmallVector<linalg::LinalgOp> linalg_ops;
  module.walk(
      [&](linalg::LinalgOp linalg_op) { linalg_ops.push_back(linalg_op); });

  IRRewriter rewriter(&getContext());
  for (auto linalg_op : linalg_ops) {
    LLVM_DEBUG(llvm::dbgs() << "  Lowering: " << linalg_op->getName() << "\n");

    rewriter.setInsertionPoint(linalg_op);

    // Determine tile sizes from output operand shape (needed for loop
    // creation)
    SmallVector<int64_t> tile_sizes = determineTileSizes(linalg_op);

    if (tile_sizes.empty()) {
      linalg_op.emitError("Could not determine tile sizes");
      signalPassFailure();
      return;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "  Tile sizes: ";
      for (auto i : tile_sizes) llvm::dbgs() << i << ", ";
      llvm::dbgs() << "\n";
    });

    // Tile the linalg.generic operation with the computed tile sizes
    linalg::LinalgTilingOptions tiling_options;
    tiling_options.setTileSizes(tile_sizes);
    tiling_options.setLoopType(linalg::LinalgTilingLoopType::Loops);

    FailureOr<linalg::TiledLinalgOp> tiled_result =
        linalg::tileLinalgOp(rewriter, linalg_op, tiling_options);

    if (failed(tiled_result)) {
      linalg_op.emitError("Failed to tile linalg operation");
      signalPassFailure();
      return;
    }

    // Replace uses of original results with tiled results
    assert(tiled_result->tensorResults.size() == linalg_op->getNumResults() &&
           "Tiled result count must match original result count");
    for (size_t i = 0; i < linalg_op->getNumResults(); ++i) {
      rewriter.replaceAllUsesWith(linalg_op->getResult(i),
                                  tiled_result->tensorResults[i]);
    }

    // Erase the original linalg operation. It has been replaced by a loop nest
    // containing the tiled linalg operation.
    rewriter.eraseOp(linalg_op);

    LLVM_DEBUG(llvm::dbgs() << "  Successfully tiled linalg operation\n");
    LLVM_DEBUG(llvm::dbgs() << "After loops created: \n" << module << "\n\n");

    // Store the tiled loops for pipeline creation
    if (!tiled_result->loops.empty()) {
      tiled_loops_.assign(tiled_result->loops.begin(),
                          tiled_result->loops.end());
      auto innermost_loop = llvm::cast<scf::ForOp>(tiled_result->loops.back());
      createPipeline(innermost_loop, tile_sizes);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "After pipeline created: \n" << module << "\n\n");

  // Replace access tiles with reinterpret_cast after pipeline creation
  replaceAccessTilesWithReinterpretCast(module);

  // Cleanup operations (removes original load/store/compute ops)
  cleanupOperations();
  LLVM_DEBUG(llvm::dbgs() << "After cleanup: \n" << module << "\n\n");
}

void ConstructThreeStagePipelinePass::createPipeline(
    scf::ForOp innermost_loop, const SmallVector<int64_t>& tile_sizes) {
  LLVM_DEBUG(llvm::dbgs() << "Creating ktdf.pipeline with three stages\n");

  compute_ops_.clear();
  innermost_loop.getBody()->walk(
      [&](linalg::LinalgOp linalg_op) { compute_ops_.push_back(linalg_op); });

  LLVM_DEBUG(llvm::dbgs() << "  Found " << compute_ops_.size()
                          << " linalg compute operations\n");

  // Save original scf.yield operands to ops_to_delete_ before changing the
  // yield This ensures tensor.insert_slice and linalg.generic get cleaned up
  // later
  auto yield_op = cast<scf::YieldOp>(innermost_loop.getBody()->getTerminator());
  for (auto operand : yield_op.getOperands()) {
    if (auto* def_op = operand.getDefiningOp()) {
      ops_to_delete_.push_back(def_op);
    }
  }

  // Fix scf.yield to yield iter_args instead of linalg results
  // This prevents issues when we delete the old linalg.generic operations
  OpBuilder yield_builder(yield_op);
  SmallVector<Value> new_yield_operands;
  for (auto iter_arg : innermost_loop.getRegionIterArgs()) {
    new_yield_operands.push_back(iter_arg);
  }
  scf::YieldOp::create(yield_builder, yield_op.getLoc(), new_yield_operands);
  yield_op.erase();

  OpBuilder builder(&getContext());

  // Set insertion point at the beginning of the loop body
  Block* loop_body = innermost_loop.getBody();
  builder.setInsertionPointToStart(loop_body);

  // Create ktdf.pipeline operation
  auto pipeline_op =
      mlir::ktdf::PipelineOp::create(builder, innermost_loop.getLoc());
  Region& pipeline_region = pipeline_op.getRegion();
  Block* pipeline_body = &pipeline_region.front();
  builder.setInsertionPointToStart(pipeline_body);

  // Create ktdf.private operation that yields FIFO slots + three tokens
  auto token_type = mlir::ktdf::TokenType::get(&getContext());
  SmallVector<Type> private_result_types;

  SmallVector<int64_t> memref_shape = determineMemrefShape(tile_sizes);

  // Add FIFO slot types for each load operation
  SmallVector<mlir::ktdf::FifoSlotType> fifo_slot_types;
  for (auto load_op : load_ops_) {
    // Get element type from load result
    auto tensor_type =
        dyn_cast<RankedTensorType>(load_op.getResult().getType());
    if (!tensor_type) {
      load_op.emitError("Expected RankedTensorType result from ktdp.load");
      signalPassFailure();
      return;
    }
    auto element_type = tensor_type.getElementType();

    // Calculate total number of elements in the tensor
    int64_t num_elements = 1;
    for (auto dim : tensor_type.getShape()) {
      num_elements *= dim;
    }

    // Create FIFO slot type: lxlu-to-sfp for loads
    auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
        &getContext(), mlir::ktdf::FifoType::LxluToSfp, num_elements,
        element_type);
    fifo_slot_types.push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Add FIFO slot types for each store operation
  for (auto store_op : store_ops_) {
    // Get element type from store data operand
    auto tensor_type =
        dyn_cast<RankedTensorType>(store_op.getDataTile().getType());
    if (!tensor_type) {
      store_op.emitError(
          "Expected RankedTensorType data operand in ktdp.store");
      signalPassFailure();
      return;
    }
    auto element_type = tensor_type.getElementType();

    // Calculate total number of elements in the tensor
    int64_t num_elements = 1;
    for (auto dim : tensor_type.getShape()) {
      num_elements *= dim;
    }

    // Create FIFO slot type: sfp-to-lxsu for stores
    auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
        &getContext(), mlir::ktdf::FifoType::SfpToLxsu, num_elements,
        element_type);
    fifo_slot_types.push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Last 3 results are tokens for the 3 stages
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);

  auto private_op = mlir::ktdf::PrivateOp::create(
      builder, innermost_loop.getLoc(), private_result_types);
  Region& private_region = private_op.getRegion();
  Block* private_body = builder.createBlock(&private_region);
  builder.setInsertionPointToStart(private_body);

  // Create ktdf.fifo.allocate for each load and store operation
  SmallVector<Value> fifo_results;
  for (auto fifo_slot_type : fifo_slot_types) {
    // FifoAllocateOp takes: builder, location, result types, operands (dynamic
    // sizes) Since we're using static sizes, operands is empty
    auto fifo_alloc_op = mlir::ktdf::FifoAllocateOp::create(
        builder, innermost_loop.getLoc(), TypeRange{fifo_slot_type},
        ValueRange{});
    fifo_results.push_back(fifo_alloc_op.getResult(0));
  }

  // Create three tokens inside ktdf.private after FIFO allocations
  auto t1 = mlir::ktdf::CreateTokenOp::create(builder, innermost_loop.getLoc(),
                                              token_type);
  auto t2 = mlir::ktdf::CreateTokenOp::create(builder, innermost_loop.getLoc(),
                                              token_type);
  auto t3 = mlir::ktdf::CreateTokenOp::create(builder, innermost_loop.getLoc(),
                                              token_type);

  // Add the three tokens at the end
  fifo_results.push_back(t1.getResult());
  fifo_results.push_back(t2.getResult());
  fifo_results.push_back(t3.getResult());

  mlir::ktdf::PrivateYieldOp::create(builder, innermost_loop.getLoc(),
                                     fifo_results);
  // Tokens are at the end: fifo_count + 0, fifo_count + 1, fifo_count + 2
  size_t fifo_count = load_ops_.size() + store_ops_.size();

  // Create three stages with sequential dependencies
  builder.setInsertionPointToEnd(pipeline_body);
  auto stage1 = mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{},
      /*depends_out=*/ValueRange{private_op.getResult(fifo_count + 0)});

  // Add data transfer operations in stage1 for loads
  Region& stage1_region = stage1.getRegion();
  Block* stage1_body = &stage1_region.front();
  builder.setInsertionPointToStart(stage1_body);
  createDataTransfers(builder, innermost_loop.getLoc(), private_op,
                      memref_shape,
                      /*is_load=*/true, /*private_result_offset=*/0);

  builder.setInsertionPointToEnd(pipeline_body);
  auto stage2 = mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{private_op.getResult(fifo_count + 0)},
      /*depends_out=*/ValueRange{private_op.getResult(fifo_count + 1)});

  // Create linalg compute operations in stage 2
  Region& stage2_region = stage2.getRegion();
  Block* stage2_body = &stage2_region.front();
  builder.setInsertionPointToStart(stage2_body);
  createComputeOps(builder, innermost_loop.getLoc(), private_op);

  builder.setInsertionPointToEnd(pipeline_body);
  auto stage3 = mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{private_op.getResult(fifo_count + 1)},
      /*depends_out=*/ValueRange{private_op.getResult(fifo_count + 2)});

  // Add data transfer operations in stage3 for stores
  Region& stage3_region = stage3.getRegion();
  Block* stage3_body = &stage3_region.front();
  builder.setInsertionPointToStart(stage3_body);
  createDataTransfers(
      builder, innermost_loop.getLoc(), private_op, memref_shape,
      /*is_load=*/false, /*private_result_offset=*/load_ops_.size());
}

void ConstructThreeStagePipelinePass::deleteOpAndUnusedAncestors(
    Operation* op) {
  // Collect operands before erasing
  SmallVector<Value> operands(op->getOperands().begin(),
                              op->getOperands().end());
  op->erase();

  // Check each operand's defining op
  for (auto operand : operands) {
    if (auto* def_op = operand.getDefiningOp()) {
      if (isa<scf::ForOp>(def_op)) continue;
      bool has_uses = false;
      for (auto result : def_op->getResults()) {
        if (!result.use_empty()) {
          has_uses = true;
          break;
        }
      }
      if (def_op->use_empty()) deleteOpAndUnusedAncestors(def_op);
    }
  }
}

void ConstructThreeStagePipelinePass::cleanupOperations() {
  LLVM_DEBUG(llvm::dbgs() << "Cleaning up operations\n");

  for (auto* op : ops_to_delete_) {
    if (!op || !op->use_empty()) continue;
    deleteOpAndUnusedAncestors(op);
  }
}

void ConstructThreeStagePipelinePass::createDataTransfers(
    OpBuilder& builder, Location loc, mlir::ktdf::PrivateOp private_op,
    ArrayRef<int64_t> memref_shape, bool is_load,
    size_t private_result_offset) {
  // Collect loop induction variables and upper bounds from tiled loops
  SmallVector<Value> loop_ivs;
  SmallVector<int64_t> loop_bounds;
  for (auto* loop_op : tiled_loops_) {
    auto for_op = llvm::cast<scf::ForOp>(loop_op);
    loop_ivs.push_back(for_op.getInductionVar());

    // Extract upper bound as constant
    auto ub_const =
        for_op.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    assert(ub_const);
    loop_bounds.push_back(ub_const.value());
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  Creating " << (is_load ? "load" : "store")
                 << " data transfers with " << loop_ivs.size() << " loop IVs\n";
  });

  // Get the appropriate operation list
  size_t op_count = is_load ? load_ops_.size() : store_ops_.size();

  const int64_t stick_size = schedulerExtContext().getStickSize();

  // Create data_transfer for each operation
  for (size_t i = 0; i < op_count; ++i) {
    // Get the access_tile value (operand of load/store op)
    Value access_tile_value;

    if (is_load) {
      auto load_op = load_ops_[i];
      // Get the operand directly - load has one operand (access_tile)
      access_tile_value = load_op.getOperand();
    } else {
      auto store_op = store_ops_[i];
      // Get the access tile operand - store has two operands (data_tile,
      // access_tile) Use getOperation()->getOperand(1) to get the second
      // operand
      access_tile_value = store_op.getOperation()->getOperand(1);
    }

    // Get the FIFO slot from private results
    Value fifo_slot = private_op.getResult(private_result_offset + i);

    // Prepare indices and sizes for access_tile
    SmallVector<Value> access_tile_indices(loop_ivs.begin(), loop_ivs.end());
    SmallVector<int64_t> access_tile_sizes(loop_bounds.begin(),
                                           loop_bounds.end());

    // FIFO slot size is stick_size
    SmallVector<int64_t> fifo_sizes = {stick_size};

    // Create data_transfer with direction based on is_load
    // Use access_tile_value which will be replaced with reinterpret_cast later
    if (is_load) {
      // Load: from access_tile to FIFO slot
      mlir::ktdf::DataTransferOp::create(builder, loc, access_tile_value,
                                         access_tile_indices, access_tile_sizes,
                                         fifo_slot, ValueRange{}, fifo_sizes);
    } else {
      // Store: from FIFO slot to access_tile
      mlir::ktdf::DataTransferOp::create(
          builder, loc, fifo_slot, ValueRange{}, fifo_sizes, access_tile_value,
          access_tile_indices, access_tile_sizes);
    }
  }
}

void ConstructThreeStagePipelinePass::runOnOperation() {
  if (DisableThisPass) return;

  ModuleOp module = getOperation();

  LLVM_DEBUG(
      llvm::dbgs() << "Starting ConstructThreeStagePipeline transformation\n");

  // Initialize const_builder_ and set insertion point at start of first
  // function
  module.walk([&](func::FuncOp func_op) {
    if (!func_op.empty()) {
      const_builder_.emplace(&getContext());
      const_builder_->setInsertionPointToStart(&func_op.front());
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Step 1: Generalize named linalg operations to linalg.generic
  if (failed(generalizeLinalgOps(module))) {
    module.emitError("Failed to generalize linalg operations");
    signalPassFailure();
    return;
  }

  // Step 2: Fuse consecutive linalg.generic operations
  if (failed(fuseLinalgOps(module))) {
    module.emitError("Failed to fuse linalg operations");
    signalPassFailure();
    return;
  }

  // Step 3: Lower linalg.generic operations to scf.for loops and create
  // pipeline
  lowerLinalgToLoops(module);
}

}  // namespace

namespace ktdf {

std::unique_ptr<mlir::Pass> createConstructThreeStagePipelinePass(
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<ConstructThreeStagePipelinePass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> createConstructThreeStagePipelinePass() {
  return std::make_unique<ConstructThreeStagePipelinePass>(
      scheduler::SchedulerExtContext::dummyContext());
}

}  // namespace ktdf

void ConstructThreeStagePipelinePass::createComputeOps(
    OpBuilder& builder, Location loc, mlir::ktdf::PrivateOp private_op) {
  LLVM_DEBUG(llvm::dbgs() << "Creating compute operations in stage 2\n");

  // Build a mapping from tensor values to read_from_fifo results
  DenseMap<Value, Value> value_map;

  // For each load operation, create ktdf.read_from_fifo and map the result
  for (size_t i = 0; i < load_ops_.size(); ++i) {
    auto load_op = load_ops_[i];
    Value fifo_slot = private_op.getResult(i);

    // Get the tensor type from the load result
    auto tensor_type =
        dyn_cast<RankedTensorType>(load_op.getResult().getType());
    if (!tensor_type) {
      load_op.emitError("Expected RankedTensorType result from ktdp.load");
      signalPassFailure();
      return;
    }

    // Create ktdf.read_from_fifo to read tensor from FIFO slot
    auto read_fifo_op = mlir::ktdf::ReadFromFifoOp::create(
        builder, loc, tensor_type, fifo_slot);

    // Map the load result to the read_from_fifo result
    value_map[load_op.getResult()] = read_fifo_op.getResult();
    LLVM_DEBUG(llvm::dbgs()
               << "  Created read_from_fifo for load " << i << "\n");
  }

  // Find all tensor.extract_slice operations that use load results
  // and map them to read_from_fifo results
  IRMapping mapper;
  for (auto load_op : load_ops_) {
    for (auto user : load_op.getResult().getUsers()) {
      if (auto extract_op = dyn_cast<tensor::ExtractSliceOp>(user)) {
        // Map extract_slice result to read_from_fifo result
        mapper.map(extract_op.getResult(), value_map[load_op.getResult()]);
        LLVM_DEBUG(llvm::dbgs()
                   << "  Mapped extract_slice to read_from_fifo\n");
      }
    }
  }

  // Clone compute operations into stage 2 and create write_to_fifo for each
  // Don't delete them here - they're already in ops_to_delete_ from the yield
  // operands
  for (size_t i = 0; i < compute_ops_.size(); ++i) {
    auto compute_op = compute_ops_[i];
    LLVM_DEBUG(llvm::dbgs()
               << "  Cloning compute op: " << compute_op->getName() << "\n");

    // Get the output tensor type from the compute operation
    auto output_type =
        dyn_cast<RankedTensorType>(compute_op->getResult(0).getType());
    if (!output_type) {
      compute_op->emitError(
          "Expected RankedTensorType result from linalg operation");
      signalPassFailure();
      return;
    }

    // Create a dummy tensor.empty for the output operand
    auto empty_tensor = tensor::EmptyOp::create(
        builder, loc, output_type.getShape(), output_type.getElementType());

    // Map the output extract_slice to the empty tensor
    // Find the output operand (last DPS output operand)
    auto linalg_op = dyn_cast<linalg::LinalgOp>(compute_op.getOperation());
    if (linalg_op && linalg_op.getNumDpsInits() > 0) {
      Value output_operand = linalg_op.getDpsInitOperand(0)->get();
      mapper.map(output_operand, empty_tensor.getResult());
    }

    // Clone the operation with mapped operands
    auto* cloned = builder.clone(*compute_op, mapper);

    // Create ktdf.write_to_fifo to write the cloned linalg result to FIFO slot
    // Assuming one compute op per store op
    if (i < store_ops_.size()) {
      Value fifo_slot = private_op.getResult(load_ops_.size() + i);
      mlir::ktdf::WriteToFifoOp::create(builder, loc, cloned->getResult(0),
                                        fifo_slot);
      LLVM_DEBUG(llvm::dbgs()
                 << "  Created write_to_fifo for store " << i << "\n");
    }
  }
}

void ConstructThreeStagePipelinePass::replaceAccessTilesWithReinterpretCast(
    ModuleOp module) {
  SmallVector<ktdp::ConstructAccessTilesOp> access_tiles;
  module.walk([&](ktdp::ConstructAccessTilesOp access_tile) {
    access_tiles.push_back(access_tile);
  });

  if (access_tiles.empty()) {
    return;
  }

  // Find the last memory view operation to insert reinterpret_casts after it
  Operation* last_memory_view = nullptr;
  module.walk([&](ktdp::ConstructMemoryViewOp memory_view) {
    last_memory_view = memory_view.getOperation();
  });

  if (!last_memory_view) {
    module.emitError("No memory view operations found");
    signalPassFailure();
    return;
  }

  OpBuilder builder(&getContext());

  for (auto access_tile : access_tiles) {
    // Get the memory view (source memref) - first operand
    Value memory_view = access_tile.getBase();
    auto memory_view_type = dyn_cast<MemRefType>(memory_view.getType());
    if (!memory_view_type) {
      access_tile.emitError("Memory view is not a memref type");
      signalPassFailure();
      return;
    }

    // Get the access tile indices - remaining operands after the first
    SmallVector<Value> indices = access_tile.getIndices();
    size_t num_indices = indices.size();

    // Get the access tile result type to determine sizes
    auto access_tile_type =
        dyn_cast<ktdp::AccessTileType>(access_tile.getResult().getType());
    if (!access_tile_type) {
      access_tile.emitError("Result is not an access tile type");
      signalPassFailure();
      return;
    }

    // Get tile dimensions from the access tile type
    auto tile_shape = access_tile_type.getShape();
    SmallVector<int64_t> tile_dims(tile_shape.begin(), tile_shape.end());

    // Get strides from memory view type
    SmallVector<int64_t> strides;
    if (auto strided_layout =
            dyn_cast<StridedLayoutAttr>(memory_view_type.getLayout())) {
      strides.assign(strided_layout.getStrides().begin(),
                     strided_layout.getStrides().end());
    } else {
      // Default strides for row-major layout
      int64_t stride = 1;
      for (int i = memory_view_type.getRank() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= memory_view_type.getShape()[i];
      }
    }

    // Set insertion point after the last memory view (before access_tiles and
    // loops)
    builder.setInsertionPointAfter(last_memory_view);
    Location loc = access_tile.getLoc();

    // TODO: Calculate proper offset as sum of (index[i] * stride[i]) for all
    // dimensions For now, just use the first index as offset
    Value offset;
    if (num_indices > 0) {
      offset = indices[0];
    } else {
      offset = arith::ConstantIndexOp::create(*const_builder_, loc, 0);
    }

    // Create sizes for reinterpret_cast
    SmallVector<OpFoldResult> sizes;
    for (auto dim : tile_dims) {
      sizes.push_back(builder.getIndexAttr(dim));
    }

    // Create strides for reinterpret_cast (same as memory view)
    SmallVector<OpFoldResult> reinterpret_strides;
    for (auto stride : strides) {
      reinterpret_strides.push_back(builder.getIndexAttr(stride));
    }

    // Create result memref type with dynamic offset and specified strides
    SmallVector<int64_t> result_shape(tile_dims.begin(), tile_dims.end());

    // Create StridedLayoutAttr with dynamic offset
    auto strided_layout =
        StridedLayoutAttr::get(builder.getContext(),
                               ShapedType::kDynamic,  // Dynamic offset
                               strides);

    auto result_type =
        MemRefType::get(result_shape, memory_view_type.getElementType(),
                        strided_layout, memory_view_type.getMemorySpace());

    // Create reinterpret_cast
    // Wrap offset Value in OpFoldResult
    OpFoldResult offset_fold_result(offset);
    auto cast_op = memref::ReinterpretCastOp::create(
        builder, loc, result_type, memory_view, offset_fold_result, sizes,
        reinterpret_strides);
    // Replace all uses of access tile with reinterpret_cast
    access_tile.replaceAllUsesWith(cast_op.getResult());

    // Mark access tile for deletion
    ops_to_delete_.push_back(access_tile.getOperation());
  }
}
