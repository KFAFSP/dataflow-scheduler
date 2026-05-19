//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
// This pass constructs a three-stage pipeline.
//
// This pass performs the following steps in order to create a three-stage
// pipeline:
//   Step 1: Generalize linalg, arith, and math operations to linalg.generic
//   Step 2: Fuse consecutive linalg.generic operations
//   Step 3: Create loops from linalg operations (tiling)
//   Step 4: Create pipeline with three stages (load, compute, store)
//   Step 5: Replace access tiles with memref.reinterpret_cast
//   Step 6: Cleanup
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpOps.hpp"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"
#include "src/Transform/Utils/CustomLinalgTiling.h"
#include "src/Transform/Utils/Utils.h"
#include "src/Utils/SchedulerExtContext.h"

#define PASS_NAME "construct-three-stage-pipeline"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace ktdf {
#define GEN_PASS_DEF_CONSTRUCTTHREESTAGEPIPELINEPASS
#include "src/Transform/Passes.h.inc"
}  // namespace ktdf

namespace {
const char VerboseDebug[] = DEBUG_TYPE "-verbose";

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
  }

  void runOnOperation() final;

 private:
  const scheduler::SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  // Process a single function
  void runOnFunc(func::FuncOp func_op);

  // Reset state between function processing
  void resetState();

  // Generalize named linalg operations to linalg.generic
  LogicalResult generalizeLinalgArithMathOps(func::FuncOp func_op);

  // Fuse consecutive linalg.generic operations
  LogicalResult fuseLinalgOps(func::FuncOp func_op);

  // Determine tile sizes based on vector_length from linalg operation
  SmallVector<int64_t> determineTileSizes(linalg::LinalgOp linalgOp);

  // Create loops from linalg operations by tiling
  void createLoopsFromLinalg(ArrayRef<linalg::LinalgOp> linalg_ops);

  // Create a 3-stage pipeline inside innermost_loop, with one stage for loads,
  // computes, stores.
  void createPipeline(scf::ForOp innermost_loop);

  // Create linalg compute operations in stage 2
  void createComputeOps(OpBuilder& builder, Location loc,
                        mlir::ktdf::PrivateOp private_op);

  // Create data transfer operations for loads or stores
  // is_load=true: transfer from access_tile to FIFO slot
  // is_load=false: transfer from FIFO slot to access_tile
  // private_result_offset: index into private_op results where FIFO slots for
  //   this transfer start.
  void createDataTransfers(OpBuilder& builder, Location loc,
                           mlir::ktdf::PrivateOp private_op,
                           ArrayRef<int64_t> tile_sizes, bool is_load,
                           size_t private_result_offset);

  // Create ktdf.private operation with FIFO slots and tokens
  // Returns the created private operation
  mlir::ktdf::PrivateOp createPrivateOp(OpBuilder& builder, Location loc);

  // Annotate loops with loop_type attributes based on linalg iterator types
  void annotateLoopsWithIteratorTypes(ArrayRef<Operation*> loops,
                                      linalg::GenericOp generic_op);

  // Delete op and recursively delete unused chain of operands
  void deleteOpAndUnusedChainOfOperands(Operation* op);

  // Replace construct_access_tile with memref.reinterpret_cast
  void replaceAccessTilesWithReinterpretCast(func::FuncOp func_op);

  // Compute offset for reinterpret_cast from indices and strides
  Value computeReinterpretCastOffset(OpBuilder& builder, Location loc,
                                     SmallVector<Value>& indices,
                                     SmallVector<int64_t>& strides);

  // Get strides from memref type, computing default row-major strides if needed
  SmallVector<int64_t> getStridesFromMemRefType(MemRefType memref_type);

  // Clean up operations after pipeline creation
  void cleanupOperations();

  // Member variables
  const scheduler::SchedulerExtContext& scheduler_ctx_;

  // Collected ktdp.load and ktdp.store operations
  SmallVector<ktdp::LoadOp> load_ops_;
  SmallVector<ktdp::StoreOp> store_ops_;
  SmallVector<linalg::LinalgOp> compute_ops_;

  // Tiled loops from linalg tiling (outermost to innermost)
  SmallVector<Operation*> tiled_loops_;

  // Tile sizes determined from linalg operation
  SmallVector<int64_t> tile_sizes_;

  // Total number of elements (product of tile_sizes_)
  int64_t total_num_elements_ = 0;

  // Operations to delete after pipeline creation
  SmallVector<Operation*> ops_to_delete_;

  // Builder for constants at function start
  std::optional<OpBuilder> const_builder_;
};

void ConstructThreeStagePipelinePass::resetState() {
  load_ops_.clear();
  store_ops_.clear();
  compute_ops_.clear();
  tiled_loops_.clear();
  tile_sizes_.clear();
  total_num_elements_ = 0;
  ops_to_delete_.clear();
  const_builder_.reset();
}

LogicalResult ConstructThreeStagePipelinePass::generalizeLinalgArithMathOps(
    func::FuncOp func_op) {
  LLVM_DEBUG(
      llvm::dbgs()
      << "Generalizing linalg, arith, and math operations to linalg.generic\n");

  SmallVector<linalg::LinalgOp> linalg_ops;

  func_op.walk([&](Operation* op) {
    // Collect linalg operations that are not already generic
    if (auto linalg_op = dyn_cast<linalg::LinalgOp>(op)) {
      if (!isa<linalg::GenericOp>(op)) {
        linalg_ops.push_back(linalg_op);
      }
    }
  });

  IRRewriter rewriter(&getContext());

  // Generalize named linalg operations
  for (linalg::LinalgOp linalg_op : linalg_ops) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Generalizing linalg op: " << linalg_op->getName() << "\n");
    rewriter.setInsertionPoint(linalg_op);

    FailureOr<linalg::GenericOp> generic_result =
        linalg::generalizeNamedOp(rewriter, linalg_op);

    if (failed(generic_result)) {
      linalg_op.emitError("Failed to generalize linalg operation");
      return failure();
    }
  }

  // Convert arith and math operations to linalg using elementwise patterns
  LLVM_DEBUG(llvm::dbgs() << "  Converting arith/math operations\n");

  RewritePatternSet patterns(&getContext());
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);

  if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
    func_op.emitError(
        "Failed to convert arith/math operations to linalg.generic");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs()
             << "  Successfully generalized compute ops to linalg.generic\n");
  return success();
}

LogicalResult ConstructThreeStagePipelinePass::fuseLinalgOps(
    func::FuncOp func_op) {
  LLVM_DEBUG(llvm::dbgs() << "Fusing consecutive linalg.generic operations\n");

  SmallVector<linalg::GenericOp> worklist;
  func_op.walk(
      [&](linalg::GenericOp generic_op) { worklist.push_back(generic_op); });

  IRRewriter rewriter(&getContext());
  while (!worklist.empty()) {
    linalg::GenericOp consumer = worklist.pop_back_val();

    // Try to fuse producer into this consumer
    for (OpOperand* operand : consumer.getDpsInputOperands()) {
      Value input = operand->get();
      auto producer = input.getDefiningOp<linalg::GenericOp>();

      if (!producer) continue;

      rewriter.setInsertionPoint(consumer);
      // TODO: we may want do non-element-wise fusion as well such as matmul
      // followed by add.
      FailureOr<linalg::ElementwiseOpFusionResult> fusion_result =
          linalg::fuseElementwiseOps(rewriter, operand);

      if (succeeded(fusion_result)) {
        LLVM_DEBUG(llvm::dbgs() << "  Successfully fused operations\n");
        auto fused_op = cast<linalg::GenericOp>(fusion_result->fusedOp);

        // Replace uses of consumer with the fused operation (this also erases
        // consumer)
        rewriter.replaceOp(consumer, fused_op->getResults());

        // Erase the original producer operation
        rewriter.eraseOp(producer);

        // Remove producer from worklist if present
        llvm::erase(worklist, producer);

        // Add the fused operation to the worklist for further fusion in the
        // next iteration.
        worklist.push_back(fused_op);
        break;
      }
    }
  }

  return success();
}

SmallVector<int64_t> ConstructThreeStagePipelinePass::determineTileSizes(
    linalg::LinalgOp linalg_op) {
  assert(linalg_op->getNumResults() == 1 &&
         "linalg op expected to have exactly one tensor result");

  ShapedType shaped_type =
      llvm::dyn_cast<ShapedType>(linalg_op->getResult(0).getType());
  assert(shaped_type && shaped_type.hasRank());

  Type elem_type = shaped_type.getElementType();
  const int64_t vector_length = schedulerExtContext().getVectorLength(
      schedulerExtContext().getComputeUnit(), elem_type);
  SmallVector<int64_t> tile_sizes;

  ArrayRef<int64_t> shape = shaped_type.getShape();
  int64_t rank = shape.size();

  // Start from rightmost dimension and multiply until we reach vector_length
  int64_t product = 1;
  int64_t covered_dims = 0;

  for (int64_t i = rank - 1; i >= 0; --i) {
    product *= shape[i];
    covered_dims++;
    if (product >= vector_length) {
      break;
    }
  }

  // Build tile sizes: 1 for uncovered dims, partial/full for covered dims
  tile_sizes.resize(rank);
  for (int64_t i = 0; i < rank - covered_dims; ++i) {
    tile_sizes[i] = 1;
  }

  // For covered dimensions, compute tile sizes that divide evenly
  int64_t remaining_product = vector_length;
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

void ConstructThreeStagePipelinePass::annotateLoopsWithIteratorTypes(
    ArrayRef<Operation*> loops, linalg::GenericOp generic_op) {
  assert(generic_op);

  // Annotate each loop with its corresponding iterator type obtained from
  // generic_op.
  const auto& iterator_types = generic_op.getIteratorTypesArray();
  for (size_t i = 0; i < loops.size() && i < iterator_types.size(); ++i) {
    auto for_op = dyn_cast<scf::ForOp>(loops[i]);
    assert(for_op);

    mlir::ktdf::LoopType loop_type;
    if (iterator_types[i] == utils::IteratorType::parallel) {
      loop_type = mlir::ktdf::LoopType::ParallelLoop;
    } else if (iterator_types[i] == utils::IteratorType::reduction) {
      loop_type = mlir::ktdf::LoopType::ReductionLoop;
    } else {
      for_op->emitError("Unsupported iterator type");
      signalPassFailure();
      return;
    }

    auto loop_type_attr =
        mlir::ktdf::LoopTypeAttr::get(&getContext(), loop_type);
    for_op->setAttr("loop_type", loop_type_attr);

    LLVM_DEBUG(llvm::dbgs() << "  Annotated loop " << i << " with loop_type: "
                            << (loop_type == mlir::ktdf::LoopType::ParallelLoop
                                    ? "parallel"
                                    : "reduction")
                            << "\n");
  }
}

void ConstructThreeStagePipelinePass::createLoopsFromLinalg(
    ArrayRef<linalg::LinalgOp> linalg_ops) {
  LLVM_DEBUG(
      llvm::dbgs() << "Creating SCF loops from linalg.generic operations\n");

  assert(linalg_ops.size() <= 1 &&
         "Currently only supporting one linalg.generic operation after fusion");

  IRRewriter rewriter(&getContext());
  for (linalg::LinalgOp linalg_op : linalg_ops) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Processing: " << linalg_op->getName() << "\n");

    rewriter.setInsertionPoint(linalg_op);

    // Determine tile sizes from output operand shape (needed for loop
    // creation)
    tile_sizes_ = determineTileSizes(linalg_op);

    if (tile_sizes_.empty()) {
      linalg_op.emitError("Could not determine tile sizes");
      signalPassFailure();
      return;
    }

    // Calculate total number of elements (product of tile_sizes_)
    total_num_elements_ = 1;
    for (int64_t dim : tile_sizes_) {
      total_num_elements_ *= dim;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Tile sizes: ";
      for (int64_t i : tile_sizes_) llvm::dbgs() << i << ", ";
      llvm::dbgs() << "\n";
      llvm::dbgs() << "  Total num elements: " << total_num_elements_ << "\n";
    });

    // Tile the linalg.generic operation with the computed tile sizes
    // Use custom tiling that doesn't create iter_args.
    linalg::LinalgTilingOptions tiling_options;
    tiling_options.setTileSizes(tile_sizes_);
    tiling_options.setLoopType(linalg::LinalgTilingLoopType::Loops);

    FailureOr<linalg::TiledLinalgOp> tiled_result =
        linalg::customTileLinalgOp(rewriter, linalg_op, tiling_options);

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

    // Annotate loops with loop_type attributes based on iterator types
    if (!tiled_result->loops.empty()) {
      auto generic_op =
          dyn_cast<linalg::GenericOp>(tiled_result->op.getOperation());
      annotateLoopsWithIteratorTypes(tiled_result->loops, generic_op);
    }

    LLVM_DEBUG(llvm::dbgs() << "  Successfully tiled linalg operation\n");

    // Store the tiled loops for later pipeline creation
    if (!tiled_result->loops.empty()) {
      tiled_loops_.assign(tiled_result->loops.begin(),
                          tiled_result->loops.end());
    }
  }
}

mlir::ktdf::PrivateOp ConstructThreeStagePipelinePass::createPrivateOp(
    OpBuilder& builder, Location loc) {
  mlir::ktdf::TokenType token_type = mlir::ktdf::TokenType::get(&getContext());
  SmallVector<Type> private_result_types;

  // Group FIFO slot types by their FIFO type to preserve ordering
  llvm::MapVector<mlir::ktdf::FifoType, SmallVector<mlir::ktdf::FifoSlotType>>
      fifo_type_groups;

  // Add FIFO slot types for load operations
  mlir::ktdf::FifoType load_fifo_type = schedulerExtContext().getFifoTypeLoad();
  for (ktdp::LoadOp load_op : load_ops_) {
    auto tensor_type =
        dyn_cast<RankedTensorType>(load_op.getResult().getType());
    if (!tensor_type) {
      load_op.emitError("Expected RankedTensorType result from ktdp.load");
      signalPassFailure();
      return nullptr;
    }
    Type element_type = tensor_type.getElementType();
    auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
        &getContext(), load_fifo_type, total_num_elements_, element_type);
    fifo_type_groups[load_fifo_type].push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Add FIFO slot types for store operations
  mlir::ktdf::FifoType store_fifo_type =
      schedulerExtContext().getFifoTypeStore();
  for (ktdp::StoreOp store_op : store_ops_) {
    auto tensor_type =
        dyn_cast<RankedTensorType>(store_op.getDataTile().getType());
    if (!tensor_type) {
      store_op.emitError(
          "Expected RankedTensorType data operand in ktdp.store");
      signalPassFailure();
      return nullptr;
    }
    Type element_type = tensor_type.getElementType();
    auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
        &getContext(), store_fifo_type, total_num_elements_, element_type);
    fifo_type_groups[store_fifo_type].push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Add three token types for the three stages
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);

  // Create ktdf.private operation
  auto private_op =
      mlir::ktdf::PrivateOp::create(builder, loc, private_result_types);
  Region& private_region = private_op.getRegion();
  Block* private_body = &private_region.front();
  builder.setInsertionPointToStart(private_body);

  // Create ktdf.fifo.allocate operations grouped by FIFO type
  // This preserves the order of load/stores within each FIFO type
  SmallVector<Value> fifo_results;
  for (auto& [fifo_type, slot_types] : fifo_type_groups) {
    // Create a single fifo.allocate with multiple results for this FIFO type
    SmallVector<Type> result_types(slot_types.begin(), slot_types.end());
    auto fifo_alloc_op = mlir::ktdf::FifoAllocateOp::create(
        builder, loc, TypeRange{result_types}, ValueRange{});

    // Add all results from this allocation to fifo_results
    for (unsigned i = 0; i < fifo_alloc_op.getNumResults(); ++i) {
      fifo_results.push_back(fifo_alloc_op.getResult(i));
    }
  }

  // Create three tokens
  auto t1 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  auto t2 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  auto t3 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  fifo_results.push_back(t1.getResult());
  fifo_results.push_back(t2.getResult());
  fifo_results.push_back(t3.getResult());

  // Yield all results
  mlir::ktdf::PrivateYieldOp::create(builder, loc, fifo_results);

  return private_op;
}

void ConstructThreeStagePipelinePass::createPipeline(
    scf::ForOp innermost_loop) {
  LLVM_DEBUG(llvm::dbgs() << "Creating ktdf.pipeline with three stages\n");

  compute_ops_.clear();
  innermost_loop.getBody()->walk(
      [&](linalg::LinalgOp linalg_op) { compute_ops_.push_back(linalg_op); });

  LLVM_DEBUG(llvm::dbgs() << "  Found " << compute_ops_.size()
                          << " linalg compute operations\n");

  // Save original scf.yield operands to ops_to_delete_ before changing the
  // yield. This ensures tensor.insert_slice and linalg.generic get cleaned up
  // later.
  auto yield_op = cast<scf::YieldOp>(innermost_loop.getBody()->getTerminator());
  for (Value operand : yield_op.getOperands()) {
    if (auto* def_op = operand.getDefiningOp()) {
      ops_to_delete_.push_back(def_op);
    }
  }

  // Fix scf.yield to yield iter_args instead of linalg results. This allows
  // tensor.insert_slice to get cleaned up (no more uses) and the dead iter args
  // will be cleaned by a dead code elimination pass.
  OpBuilder yield_builder(yield_op);
  SmallVector<Value> new_yield_operands;
  for (BlockArgument iter_arg : innermost_loop.getRegionIterArgs()) {
    new_yield_operands.push_back(iter_arg);
  }
  scf::YieldOp::create(yield_builder, yield_op.getLoc(), new_yield_operands);
  yield_op.erase();

  OpBuilder builder(&getContext());

  // Create ktdf.pipeline operation at the start of the loop body.
  Block* loop_body = innermost_loop.getBody();
  builder.setInsertionPointToStart(loop_body);

  auto pipeline_op =
      mlir::ktdf::PipelineOp::create(builder, innermost_loop.getLoc());
  Region& pipeline_region = pipeline_op.getRegion();
  Block* pipeline_body = &pipeline_region.front();
  builder.setInsertionPointToStart(pipeline_body);

  // Create ktdf.private operation with FIFO slots and tokens
  auto private_op = createPrivateOp(builder, innermost_loop.getLoc());
  assert(private_op && "Private op creation failed");

  // Query the interface type from memory to the compute unit
  std::string interface_type =
      schedulerExtContext().getInterfaceMemToComputeUnit(
          schedulerExtContext().getComputeUnit());
  assert(interface_type == "FIFO" && "Expected FIFO interface type");

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
  createDataTransfers(builder, innermost_loop.getLoc(), private_op, tile_sizes_,
                      /*is_load=*/true, /*private_result_offset=*/0);

  builder.setInsertionPointToEnd(pipeline_body);
  auto stage2 = mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{private_op.getResult(fifo_count + 0)},
      /*depends_out=*/ValueRange{private_op.getResult(fifo_count + 1)});

  // Set applicable_units attribute for stage2 (compute stage)
  auto compute_unit = schedulerExtContext().getComputeUnit();
  SmallVector<mlir::ktdf::ComponentType> units;
  units.push_back(compute_unit);
  auto applicable_units_attr =
      mlir::ktdf::ApplicableUnitsCustomAttr::get(builder.getContext(), units);
  stage2->setAttr("applicable_units", applicable_units_attr);

  // Create read_from_fifos, compute operations, write_to_fifos in stage 2
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
  createDataTransfers(builder, innermost_loop.getLoc(), private_op, tile_sizes_,
                      /*is_load=*/false,
                      /*private_result_offset=*/load_ops_.size());
}

void ConstructThreeStagePipelinePass::deleteOpAndUnusedChainOfOperands(
    Operation* op) {
  // Collect operands before erasing
  SmallVector<Value> operands(op->getOperands().begin(),
                              op->getOperands().end());
  op->erase();

  // Recursively delete unused operand operations
  for (Value operand : operands) {
    if (auto* def_op = operand.getDefiningOp()) {
      if (def_op->use_empty()) deleteOpAndUnusedChainOfOperands(def_op);
    }
  }
}

void ConstructThreeStagePipelinePass::cleanupOperations() {
  LLVM_DEBUG(llvm::dbgs() << "Cleaning up operations\n");

  for (auto* op : ops_to_delete_) {
    if (!op || !op->use_empty()) continue;
    deleteOpAndUnusedChainOfOperands(op);
  }
}

void ConstructThreeStagePipelinePass::createDataTransfers(
    OpBuilder& builder, Location loc, mlir::ktdf::PrivateOp private_op,
    ArrayRef<int64_t> tile_sizes, bool is_load, size_t private_result_offset) {
  // Collect loop induction variables from tiled loops
  SmallVector<Value> loop_ivs;
  for (auto* loop_op : tiled_loops_) {
    auto for_op = llvm::cast<scf::ForOp>(loop_op);
    loop_ivs.push_back(for_op.getInductionVar());
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  Creating " << (is_load ? "load" : "store")
                 << " data transfers with " << loop_ivs.size() << " loop IVs\n";
  });

  // Get the appropriate operation list
  size_t op_count = is_load ? load_ops_.size() : store_ops_.size();

  // FIFO slot size is the product of tile_sizes (i.e. total_num_elements_)
  SmallVector<int64_t> fifo_sizes = {total_num_elements_};

  assert(total_num_elements_ <= schedulerExtContext().getMaxFifoSlotSize(
                                    schedulerExtContext().getComputeUnit()) &&
         "FIFO slot size exceeds maximum allowed size for compute unit");

  // Create data_transfer for each operation
  for (size_t i = 0; i < op_count; ++i) {
    // Get the access_tile value (operand of load/store op)
    Value access_tile_value;

    if (is_load) {
      ktdp::LoadOp load_op = load_ops_[i];
      access_tile_value = load_op.getOperand();
    } else {
      ktdp::StoreOp store_op = store_ops_[i];
      access_tile_value = store_op.getOperation()->getOperand(1);
    }

    // Get the FIFO slot from ktdf.private results
    Value fifo_slot = private_op.getResult(private_result_offset + i);

    // Prepare indices and sizes for access_tile
    SmallVector<Value> access_tile_indices(loop_ivs.begin(), loop_ivs.end());
    SmallVector<int64_t> access_tile_sizes(tile_sizes.begin(),
                                           tile_sizes.end());

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

SmallVector<int64_t> ConstructThreeStagePipelinePass::getStridesFromMemRefType(
    MemRefType memref_type) {
  SmallVector<int64_t> strides;
  if (auto strided_layout =
          dyn_cast<StridedLayoutAttr>(memref_type.getLayout())) {
    strides.assign(strided_layout.getStrides().begin(),
                   strided_layout.getStrides().end());
  } else {
    // Compute default row-major strides
    int64_t stride = 1;
    for (int i = memref_type.getRank() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), stride);
      stride *= memref_type.getShape()[i];
    }
  }
  return strides;
}

}  // namespace

void ConstructThreeStagePipelinePass::createComputeOps(
    OpBuilder& builder, Location loc, mlir::ktdf::PrivateOp private_op) {
  LLVM_DEBUG(llvm::dbgs() << "Creating compute operations in stage 2\n");

  if (compute_ops_.size() != 1) {
    emitError(loc, "Expected exactly one compute op after fusion, got ")
        << compute_ops_.size();
    signalPassFailure();
    return;
  }

  // Get the tiled tensor type from the compute operation
  auto tiled_tensor_type =
      dyn_cast<RankedTensorType>(compute_ops_[0]->getResult(0).getType());
  if (!tiled_tensor_type) {
    compute_ops_[0]->emitError(
        "Expected RankedTensorType result from linalg operation");
    signalPassFailure();
    return;
  }

  // Build a mapping from load results to read_from_fifo results
  DenseMap<Value, Value> value_map;

  // For each load operation, create ktdf.read_from_fifo to read tiled tensor
  // from FIFO slot, and map the load result to it.
  for (size_t i = 0; i < load_ops_.size(); ++i) {
    ktdp::LoadOp load_op = load_ops_[i];
    Value fifo_slot = private_op.getResult(i);

    auto read_fifo_op = mlir::ktdf::ReadFromFifoOp::create(
        builder, loc, tiled_tensor_type, fifo_slot);
    value_map[load_op.getResult()] = read_fifo_op.getResult();
  }

  // Find all tensor.extract_slice operations that use load results
  // and map them to corresponding read_from_fifo result.
  IRMapping mapper;
  for (ktdp::LoadOp load_op : load_ops_) {
    for (Operation* user : load_op.getResult().getUsers()) {
      if (auto extract_op = dyn_cast<tensor::ExtractSliceOp>(user)) {
        mapper.map(extract_op.getResult(), value_map[load_op.getResult()]);
      }
    }
  }

  // Clone compute operation into stage 2 and create write_to_fifo for each
  // store
  linalg::LinalgOp compute_op = compute_ops_[0];
  LLVM_DEBUG(llvm::dbgs() << "  Cloning compute op: " << compute_op->getName()
                          << "\n");

  // Create a dummy tensor.empty for the output operand with the tiled tensor
  // type
  auto empty_tensor =
      tensor::EmptyOp::create(builder, loc, tiled_tensor_type.getShape(),
                              tiled_tensor_type.getElementType());

  // Map the output extract_slice to the empty tensor
  auto linalg_op = dyn_cast<linalg::LinalgOp>(compute_op.getOperation());
  if (linalg_op && linalg_op.getNumDpsInits() > 0) {
    Value output_operand = linalg_op.getDpsInitOperand(0)->get();
    mapper.map(output_operand, empty_tensor.getResult());
  }

  auto* cloned = builder.clone(*compute_op, mapper);

  // Create ktdf.write_to_fifo for each store operation
  for (size_t i = 0; i < store_ops_.size(); ++i) {
    Value fifo_slot = private_op.getResult(load_ops_.size() + i);
    mlir::ktdf::WriteToFifoOp::create(builder, loc, cloned->getResult(0),
                                      fifo_slot);
  }
}

Value ConstructThreeStagePipelinePass::computeReinterpretCastOffset(
    OpBuilder& builder, Location loc, SmallVector<Value>& indices,
    SmallVector<int64_t>& strides) {
  // Calculate offset from access tile indices and memory view strides.
  // For an access tile %A_view[%idx0, %idx1, ...] with strides [stride0,
  // stride1, ...], the offset is: %idx0 * stride0 + %idx1 * stride1 + ...
  // This is computed using a sequence of arith.muli and arith.addi operations.

  size_t num_indices = indices.size();

  if (num_indices == 0) {
    // No indices means offset is 0
    return arith::ConstantIndexOp::create(builder, loc, 0);
  }

  // Compute terms: indices[i] * strides[i] for each dimension
  // Optimizations:
  // - Skip multiplication if stride is 1
  // - Skip addition if index is constant 0
  SmallVector<Value> terms;
  for (size_t i = 0; i < num_indices; ++i) {
    if (scheduler::utils::isTargetConstant(0, indices[i])) {
      continue;
    }

    Value term;
    if (strides[i] == 1) {
      term = indices[i];
    } else {
      Value stride_const =
          arith::ConstantIndexOp::create(builder, loc, strides[i]);
      term = arith::MulIOp::create(builder, loc, indices[i], stride_const);
    }
    terms.push_back(term);
  }

  if (terms.empty()) {
    // All indices were constant 0
    return arith::ConstantIndexOp::create(builder, loc, 0);
  } else if (terms.size() == 1) {
    // Only one term, no addition needed
    return terms[0];
  } else {
    // Add all terms together
    Value offset = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
      offset = arith::AddIOp::create(builder, loc, offset, terms[i]);
    }
    return offset;
  }
}

void ConstructThreeStagePipelinePass::replaceAccessTilesWithReinterpretCast(
    func::FuncOp func_op) {
  SmallVector<ktdp::ConstructAccessTilesOp> access_tiles;
  func_op.walk([&](ktdp::ConstructAccessTilesOp access_tile) {
    access_tiles.push_back(access_tile);
  });

  if (access_tiles.empty()) {
    return;
  }

  // Find the insertion point: right before the first loop in tiled_loops_
  // to ensure def before use
  Operation* insertion_point = nullptr;
  if (!tiled_loops_.empty()) {
    insertion_point = tiled_loops_.front();
  } else {
    // Fallback: find the last memory view operation
    func_op.walk([&](ktdp::ConstructMemoryViewOp memory_view) {
      insertion_point = memory_view.getOperation();
    });
  }

  if (!insertion_point) {
    func_op.emitError(
        "No insertion point found for reinterpret_cast operations");
    signalPassFailure();
    return;
  }

  OpBuilder builder(&getContext());

  for (ktdp::ConstructAccessTilesOp access_tile : access_tiles) {
    // Get the memory view (source memref) - first operand
    Value memory_view = access_tile.getBase();
    auto memory_view_type = dyn_cast<MemRefType>(memory_view.getType());
    if (!memory_view_type) {
      access_tile.emitError("Memory view is not a memref type");
      signalPassFailure();
      return;
    }

    // Get the access tile indices and base_map. base_map projects the index
    // operands onto the per-dimension coordinates of the source memref, so
    // indices.size() == base_map.getNumInputs() and base_map.getNumResults()
    // == memref rank. For the common case base_map is identity.
    SmallVector<Value> raw_indices = access_tile.getIndices();
    AffineMap base_map = access_tile.getBaseMap();

    // Get the access tile result type to determine sizes
    auto access_tile_type =
        dyn_cast<ktdp::AccessTileType>(access_tile.getResult().getType());
    if (!access_tile_type) {
      access_tile.emitError("Result is not an access tile type");
      signalPassFailure();
      return;
    }

    // Get tile dimensions from the access tile type
    ArrayRef<int64_t> tile_shape = access_tile_type.getShape();
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

    builder.setInsertionPoint(insertion_point);
    Location loc = access_tile.getLoc();

    // Move the memory view operation right before the insertion point to ensure
    // it dominates the reinterpret_cast
    memory_view.getDefiningOp()->moveBefore(insertion_point);

    // Apply base_map to materialize one index per source-memref dimension.
    // Operands to expandAffineMap are dim-values followed by symbol-values; the
    // op's symbol_operands feed both base_map symbols (if any) and the
    // access_tile_set, so they are appended after the raw indices.
    SmallVector<Value> expand_operands(raw_indices.begin(), raw_indices.end());
    ValueRange symbol_operands = access_tile.getSymbolOperands();
    expand_operands.append(symbol_operands.begin(), symbol_operands.end());
    std::optional<SmallVector<Value, 8>> per_dim_indices =
        affine::expandAffineMap(builder, loc, base_map, expand_operands);
    if (!per_dim_indices) {
      access_tile.emitError("Failed to expand base_map into per-dimension "
                            "indices");
      signalPassFailure();
      return;
    }
    SmallVector<Value> indices(per_dim_indices->begin(),
                               per_dim_indices->end());

    // After base_map expansion, indices.size() == memref rank == strides.size()
    if (indices.size() != strides.size()) {
      access_tile.emitError("Number of indices (")
          << indices.size() << ") does not match number of strides ("
          << strides.size() << ")";
      signalPassFailure();
      return;
    }

    // Calculate offset from per-dim indices and memory view strides
    Value offset = computeReinterpretCastOffset(builder, loc, indices, strides);

    // Create sizes for reinterpret_cast
    SmallVector<OpFoldResult> sizes;
    for (int64_t dim : tile_dims) {
      sizes.push_back(builder.getIndexAttr(dim));
    }

    // Create strides for reinterpret_cast (same as memory view)
    SmallVector<OpFoldResult> reinterpret_strides;
    for (int64_t stride : strides) {
      reinterpret_strides.push_back(builder.getIndexAttr(stride));
    }

    // Get ktdp memory space from construct_memory_view operation attribute
    auto construct_mem_view =
        dyn_cast<ktdp::ConstructMemoryViewOp>(memory_view.getDefiningOp());
    assert(construct_mem_view &&
           "Memory view must be defined by construct_memory_view operation");
    Attribute ktdp_memory_space = construct_mem_view.getMemorySpaceAttr();

    // Convert ktdp.spyre_memory_space to ktdf.memory_space
    Attribute ktdf_memory_space =
        scheduler::utils::convertKtdpToKtdfMemorySpace(builder,
                                                       ktdp_memory_space);
    if (!ktdf_memory_space) {
      access_tile.emitError(
          "Failed to convert ktdp memory space to ktdf memory space");
      signalPassFailure();
      return;
    }

    // Create a memref.memory_space_cast to add the KTDF memory space
    // This propagates the memory space from construct_mem_view to the
    // reinterpret_cast
    MemRefType cast_source_type = MemRefType::get(
        memory_view_type.getShape(), memory_view_type.getElementType(),
        memory_view_type.getLayout(), ktdf_memory_space);
    auto memory_space_cast = memref::MemorySpaceCastOp::create(
        builder, loc, cast_source_type, memory_view);

    SmallVector<int64_t> result_shape(tile_dims.begin(), tile_dims.end());
    StridedLayoutAttr strided_layout = StridedLayoutAttr::get(
        builder.getContext(), ShapedType::kDynamic, strides);
    MemRefType result_type =
        MemRefType::get(result_shape, memory_view_type.getElementType(),
                        strided_layout, ktdf_memory_space);

    OpFoldResult offset_fold_result(offset);
    auto cast_op = memref::ReinterpretCastOp::create(
        builder, loc, result_type, memory_space_cast.getResult(),
        offset_fold_result, sizes, reinterpret_strides);
    // Replace access tile with reinterpret_cast
    access_tile.replaceAllUsesWith(cast_op.getResult());
    ops_to_delete_.push_back(access_tile.getOperation());
  }
}

void ConstructThreeStagePipelinePass::runOnFunc(func::FuncOp func_op) {
  LLVM_DEBUG(llvm::dbgs() << "Processing function: " << func_op.getName()
                          << "\n");

  // Initialize const_builder_ at start of function
  if (!func_op.empty()) {
    const_builder_.emplace(&getContext());
    const_builder_->setInsertionPointToStart(&func_op.front());
  }

  // Step 1: Generalize named linalg operations and arith/math operations to
  // linalg.generic
  if (failed(generalizeLinalgArithMathOps(func_op))) {
    func_op.emitError("Failed to generalize linalg operations");
    signalPassFailure();
    return;
  }

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After generalization:\n"
                                             << func_op << "\n\n");

  // Step 2: Fuse consecutive linalg.generic operations
  if (failed(fuseLinalgOps(func_op))) {
    func_op.emitError("Failed to fuse linalg operations");
    signalPassFailure();
    return;
  }

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After fusion:\n"
                                             << func_op << "\n\n");

  // Collect ktdp load/store operations and linalg operations
  SmallVector<linalg::LinalgOp> linalg_ops;
  func_op.walk([&](Operation* op) {
    if (auto load_op = dyn_cast<ktdp::LoadOp>(op)) {
      load_ops_.push_back(load_op);
      ops_to_delete_.push_back(op);
    } else if (auto store_op = dyn_cast<ktdp::StoreOp>(op)) {
      store_ops_.push_back(store_op);
      ops_to_delete_.push_back(op);
    } else if (auto linalg_op = dyn_cast<linalg::LinalgOp>(op)) {
      linalg_ops.push_back(linalg_op);
    }
  });

  // Step 3: Create loops from linalg operations
  createLoopsFromLinalg(linalg_ops);

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After loops created:\n"
                                             << func_op << "\n\n");

  // Step 4: Create pipeline if we have tiled loops
  if (!tiled_loops_.empty()) {
    auto innermost_loop = llvm::cast<scf::ForOp>(tiled_loops_.back());
    createPipeline(innermost_loop);
  }

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After pipeline created:\n"
                                             << func_op << "\n\n");

  // Step 5: Replace access tiles with reinterpret_cast after pipeline creation
  replaceAccessTilesWithReinterpretCast(func_op);

  // Step 6: Cleanup operations (removes original load/store/compute ops)
  cleanupOperations();

  DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << "After cleanup:\n"
                                             << func_op << "\n\n");
}

void ConstructThreeStagePipelinePass::runOnOperation() {
  if (DisableThisPass) return;

  ModuleOp module = getOperation();

  LLVM_DEBUG(
      llvm::dbgs() << "Starting ConstructThreeStagePipeline transformation\n");

  // Process each function in each nested module
  module.walk([&](func::FuncOp func_op) {
    resetState();
    runOnFunc(func_op);
  });
}

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
