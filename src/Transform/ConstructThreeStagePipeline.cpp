//===------------------- ConstructThreeStagePipeline.cpp -----------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
/// This pass constructs a three-stage pipeline.
///
/// Transforms linalg operations on tensors into nested scf.for loops
/// operating on memref buffers.
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "Ktdp/KtdpTypes.hpp"
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

  // Builder for constants at function start (initialized in runOnOperation)
  std::optional<OpBuilder> const_builder_;

  // Run one-shot bufferization on the module
  LogicalResult bufferizeModule(ModuleOp module);

  // Determine tile sizes based on stick_size from linalg operation
  SmallVector<int64_t> determineTileSizes(linalg::LinalgOp linalgOp);

  // Transform linalg operations to loops
  void lowerLinalgToLoops(ModuleOp module);

  // Create ktdf.pipeline with ktdf.private and ktdf.stage operations
  void createPipeline(scf::ForOp innermost_loop, ArrayRef<int64_t> tile_sizes);

  // Create linalg compute operations in stage 2
  void createComputeOps(OpBuilder& builder, Location loc,
                        mlir::ktdf::PrivateOp private_op);

  // Create data transfer operations for loads or stores
  // is_load=true: transfer from access_tile to private memref
  // is_load=false: transfer from private memref to access_tile
  void createDataTransfers(OpBuilder& builder, Location loc,
                           mlir::ktdf::PrivateOp private_op,
                           ArrayRef<int64_t> memref_shape, bool is_load,
                           size_t private_result_offset);

  // Delete operation and recursively delete unused ancestors
  void deleteOpAndUnusedAncestors(Operation* op);

  // Replace construct_access_tile with memref.reinterpret_cast
  void replaceAccessTilesWithReinterpretCast(ModuleOp module);

  // Cleanup operations after pipeline creation
  void cleanupOperations();
};

LogicalResult ConstructThreeStagePipelinePass::bufferizeModule(
    ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Running one-shot bufferization\n");

  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.allowUnknownOps = true;  // Allow ktdp operations to pass through

  bufferization::BufferizationState state;
  if (failed(bufferization::runOneShotBufferize(module, options, state))) {
    return failure();
  }

  return success();
}

SmallVector<int64_t> ConstructThreeStagePipelinePass::determineTileSizes(
    linalg::LinalgOp linalgOp) {
  const int64_t stick_size = schedulerExtContext().getStickSize();
  SmallVector<int64_t> tileSizes;

  // Get shape from output operand (after bufferization, linalg ops have no
  // results)
  ShapedType shapedType;
  if (linalgOp->getNumResults() > 0) {
    // Tensor-based (before bufferization)
    shapedType = llvm::dyn_cast<ShapedType>(linalgOp->getResult(0).getType());
  } else if (linalgOp.getNumDpsInits() > 0) {
    // Memref-based (after bufferization) - get from output operand
    shapedType = llvm::dyn_cast<ShapedType>(
        linalgOp.getDpsInitOperand(0)->get().getType());
  }

  if (!shapedType || !shapedType.hasRank()) {
    return tileSizes;
  }

  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t rank = shape.size();

  // Start from rightmost dimension and multiply until we reach stick_size
  int64_t product = 1;
  int64_t coveredDims = 0;

  for (int64_t i = rank - 1; i >= 0; --i) {
    product *= shape[i];
    coveredDims++;
    if (product >= stick_size) {
      break;
    }
  }

  // Build tile sizes: 1 for uncovered dims, partial/full for covered dims
  tileSizes.resize(rank);
  for (int64_t i = 0; i < rank - coveredDims; ++i) {
    tileSizes[i] = 1;
  }

  // For covered dimensions, compute tile sizes that divide evenly
  int64_t remainingProduct = stick_size;
  for (int64_t i = rank - 1; i >= rank - coveredDims; --i) {
    int64_t dimSize = shape[i];
    int64_t tileSize = std::min(dimSize, remainingProduct);

    // Find largest divisor of dimSize that fits in remainingProduct
    while (tileSize > 0 &&
           (dimSize % tileSize != 0 || remainingProduct % tileSize != 0)) {
      tileSize--;
    }
    if (tileSize == 0) tileSize = 1;

    tileSizes[i] = tileSize;
    remainingProduct /= tileSize;
    if (remainingProduct <= 1) {
      remainingProduct = 1;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    Shape: [";
    for (int64_t i = 0; i < rank; ++i) {
      llvm::dbgs() << shape[i];
      if (i < rank - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n    Tile sizes: [";
    for (int64_t i = 0; i < rank; ++i) {
      llvm::dbgs() << tileSizes[i];
      if (i < rank - 1) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
  });

  return tileSizes;
}

void ConstructThreeStagePipelinePass::lowerLinalgToLoops(ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering linalg operations to loops\n");

  SmallVector<linalg::LinalgOp> linalg_ops;

  module.walk([&](Operation* op) {
    if (auto linalg_op = dyn_cast<linalg::LinalgOp>(op)) {
      linalg_ops.push_back(linalg_op);
    } else if (auto load_op = dyn_cast<ktdp::LoadOp>(op)) {
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

  IRRewriter rewriter(&getContext());
  for (auto linalg_op : linalg_ops) {
    if (!linalg_op->getBlock()) continue;

    LLVM_DEBUG(llvm::dbgs() << "  Lowering: " << linalg_op->getName() << "\n");

    rewriter.setInsertionPoint(linalg_op);

    // Determine tile sizes from output operand shape
    SmallVector<int64_t> tile_sizes = determineTileSizes(linalg_op);

    if (tile_sizes.empty()) {
      linalg_op.emitError("Could not determine tile sizes");
      signalPassFailure();
      return;
    }
    llvm::dbgs() << "Tilesizes ";
    for (auto i : tile_sizes) llvm::dbgs() << i << ", ";
    llvm::dbgs() << "\n";
    // Configure tiling options
    linalg::LinalgTilingOptions options;
    options.setTileSizes(tile_sizes);
    options.setLoopType(linalg::LinalgTilingLoopType::Loops);

    // Apply tiling
    FailureOr<linalg::TiledLinalgOp> tiled_result =
        linalg::tileLinalgOp(rewriter, linalg_op, options);

    if (failed(tiled_result)) {
      linalg_op.emitError("Failed to tile linalg operation");
      signalPassFailure();
      return;
    }

    // Replace original op with tiled version
    rewriter.replaceOp(linalg_op, tiled_result->tensorResults);

    // Store tiled loops and create pipeline
    if (!tiled_result->loops.empty()) {
      tiled_loops_.assign(tiled_result->loops.begin(),
                          tiled_result->loops.end());
      auto innermost_loop = llvm::cast<scf::ForOp>(tiled_result->loops.back());
      createPipeline(innermost_loop, tile_sizes);
    }
  }

  // Replace access tiles with reinterpret_cast
  replaceAccessTilesWithReinterpretCast(module);

  // Cleanup operations (removes original load/store/compute ops)
  cleanupOperations();
}

void ConstructThreeStagePipelinePass::createPipeline(
    scf::ForOp innermost_loop, ArrayRef<int64_t> tile_sizes) {
  LLVM_DEBUG(llvm::dbgs() << "Creating ktdf.pipeline with three stages\n");

  compute_ops_.clear();
  innermost_loop.getBody()->walk([&](linalg::LinalgOp linalg_op) {
    compute_ops_.push_back(linalg_op);
    ops_to_delete_.push_back(linalg_op.getOperation());
  });

  LLVM_DEBUG(llvm::dbgs() << "  Found " << compute_ops_.size()
                          << " linalg compute operations\n");

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

  // Create ktdf.private operation that yields memref allocs + three tokens
  auto token_type = mlir::ktdf::TokenType::get(&getContext());
  SmallVector<Type> private_result_types;

  // Compute memref shape from tile sizes (drop leading 1s)
  SmallVector<int64_t> memref_shape;
  for (int64_t tile_size : tile_sizes) {
    if (tile_size != 1 || !memref_shape.empty()) {
      memref_shape.push_back(tile_size);
    }
  }
  // If all tile sizes were 1, keep at least one
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

  // Add memref types for each load operation
  SmallVector<MemRefType> alloc_types;
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

    auto memref_type = MemRefType::get(
        memref_shape, element_type, MemRefLayoutAttrInterface{},
        schedulerExtContext().getTransferMemSpace(&getContext(), "SFP"));
    alloc_types.push_back(memref_type);
    private_result_types.push_back(memref_type);
  }

  // Add memref types for each store operation
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

    auto memref_type = MemRefType::get(
        memref_shape, element_type, MemRefLayoutAttrInterface{},
        schedulerExtContext().getTransferMemSpace(&getContext(), "SFP"));
    alloc_types.push_back(memref_type);
    private_result_types.push_back(memref_type);
  }

  // Last 3 results are tokens for the 3 stages
  private_result_types.push_back(token_type);  // Stage 1 token
  private_result_types.push_back(token_type);  // Stage 2 token
  private_result_types.push_back(token_type);  // Stage 3 token

  auto private_op = mlir::ktdf::PrivateOp::create(
      builder, innermost_loop.getLoc(), private_result_types);
  Region& private_region = private_op.getRegion();
  Block* private_body = builder.createBlock(&private_region);
  builder.setInsertionPointToStart(private_body);

  // Create three tokens inside ktdf.private
  auto t1 = mlir::ktdf::CreateTokenOp::create(builder, innermost_loop.getLoc(),
                                              token_type);
  auto t2 = mlir::ktdf::CreateTokenOp::create(builder, innermost_loop.getLoc(),
                                              token_type);
  auto t3 = mlir::ktdf::CreateTokenOp::create(builder, innermost_loop.getLoc(),
                                              token_type);

  // Create memref.alloc for each load and store operation
  SmallVector<Value> alloc_results;
  for (auto memref_type : alloc_types) {
    auto alloc_op =
        memref::AllocOp::create(builder, innermost_loop.getLoc(), memref_type);
    alloc_results.push_back(alloc_op.getResult());
  }

  // Add the three tokens at the end
  alloc_results.push_back(t1.getResult());
  alloc_results.push_back(t2.getResult());
  alloc_results.push_back(t3.getResult());

  // Yield allocated memrefs and then the three tokens
  mlir::ktdf::PrivateYieldOp::create(builder, innermost_loop.getLoc(),
                                     alloc_results);

  // Back to pipeline body to create stages
  builder.setInsertionPointToEnd(pipeline_body);

  // Tokens are at the end: memref_count + 0, memref_count + 1, memref_count + 2
  size_t memref_count = load_ops_.size() + store_ops_.size();

  // Create three stages with sequential dependencies
  auto stage1 = mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{},
      /*depends_out=*/ValueRange{private_op.getResult(memref_count + 0)});

  // Add data transfer operations in stage1 for loads
  Region& stage1_region = stage1.getRegion();
  Block* stage1_body = &stage1_region.front();
  builder.setInsertionPointToStart(stage1_body);
  createDataTransfers(builder, innermost_loop.getLoc(), private_op,
                      memref_shape,
                      /*is_load=*/true, /*private_result_offset=*/0);

  // Back to pipeline body for stage 2
  builder.setInsertionPointToEnd(pipeline_body);

  auto stage2 = mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{private_op.getResult(memref_count + 0)},
      /*depends_out=*/ValueRange{private_op.getResult(memref_count + 1)});

  // Create linalg compute operations in stage 2
  Region& stage2_region = stage2.getRegion();
  Block* stage2_body = &stage2_region.front();
  builder.setInsertionPointToStart(stage2_body);
  createComputeOps(builder, innermost_loop.getLoc(), private_op);

  // Back to pipeline body for stage 3
  builder.setInsertionPointToEnd(pipeline_body);

  auto stage3 = mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{private_op.getResult(memref_count + 1)},
      /*depends_out=*/ValueRange{private_op.getResult(memref_count + 2)});

  // Add data transfer operations in stage3 for stores
  Region& stage3_region = stage3.getRegion();
  Block* stage3_body = &stage3_region.front();
  builder.setInsertionPointToStart(stage3_body);
  createDataTransfers(
      builder, innermost_loop.getLoc(), private_op, memref_shape,
      /*is_load=*/false, /*private_result_offset=*/load_ops_.size());
  // Cleanup operations (remove original linalg ops and ktdp load/store ops)
  // cleanupOperations();
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
      // Check if this operation has no uses
      bool has_uses = false;
      for (auto result : def_op->getResults()) {
        if (!result.use_empty()) {
          has_uses = true;
          break;
        }
      }

      if (def_op->use_empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  Deleting unused ancestor: "
                                << def_op->getName() << "\n");
        deleteOpAndUnusedAncestors(def_op);
      }
    }
  }
}

void ConstructThreeStagePipelinePass::cleanupOperations() {
  LLVM_DEBUG(llvm::dbgs() << "Cleaning up operations\n");

  for (auto* op : ops_to_delete_) {
    if (!op || !op->use_empty()) continue;
    // Get the name before deleting to avoid use-after-free
    LLVM_DEBUG({
      auto op_name = op->getName();
      llvm::dbgs() << "  Deleting: " << op_name << "\n";
    });
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

  // Create constant 0 for private memref index (reuse from function start)
  auto c0 = arith::ConstantIndexOp::create(*const_builder_, loc, 0);

  // Get the appropriate operation list
  size_t op_count = is_load ? load_ops_.size() : store_ops_.size();

  // Create data_transfer for each operation
  for (size_t i = 0; i < op_count; ++i) {
    // Get the access_tile value (result of construct_access_tile)
    // This will later be replaced with reinterpret_cast result via
    // replaceAllUsesWith
    Value access_tile_value;

    if (is_load) {
      auto load_op = load_ops_[i];
      access_tile_value = load_op.getAccessTile();
    } else {
      auto store_op = store_ops_[i];
      access_tile_value = store_op.getAccessTile();
    }

    // Get the allocated memref from private results
    Value private_memref = private_op.getResult(private_result_offset + i);

    // Prepare indices and sizes
    SmallVector<Value> access_tile_indices(loop_ivs.begin(), loop_ivs.end());
    SmallVector<int64_t> private_sizes(memref_shape.begin(),
                                       memref_shape.end());

    // Access tile sizes are the loop bounds (e.g., [1, 64])
    SmallVector<int64_t> access_tile_sizes(loop_bounds.begin(),
                                           loop_bounds.end());

    // Create data_transfer with direction based on is_load
    // Use access_tile_value which will be replaced with reinterpret_cast later
    if (is_load) {
      // Load: from access_tile to private
      mlir::ktdf::DataTransferOp::create(
          builder, loc, access_tile_value, access_tile_indices,
          access_tile_sizes, private_memref, ValueRange{c0}, private_sizes);
    } else {
      // Store: from private to access_tile
      mlir::ktdf::DataTransferOp::create(
          builder, loc, private_memref, ValueRange{c0}, private_sizes,
          access_tile_value, access_tile_indices, access_tile_sizes);
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

  // Step 1: Bufferize the module (convert tensors to memrefs)
  if (failed(bufferizeModule(module))) {
    module.emitError("Failed to bufferize module");
    signalPassFailure();
    return;
  }

  // Step 2: Lower linalg operations to scf.for loops and create pipeline
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

  // Build a mapping from bufferization.to_memref results to private memref
  // results
  DenseMap<Value, Value> value_map;

  // Map load operations to private memrefs (starting at index 0)
  for (size_t i = 0; i < load_ops_.size(); ++i) {
    // Find the bufferization.to_memref that uses this load
    bool found = false;
    for (auto user : load_ops_[i].getResult().getUsers()) {
      if (auto to_memref = dyn_cast<bufferization::ToBufferOp>(user)) {
        value_map[to_memref.getResult()] = private_op.getResult(i);
        found = true;
        LLVM_DEBUG(llvm::dbgs()
                   << "  Mapped load " << i << " to private#" << i << "\n");
        break;
      }
    }
    assert(found &&
           "Failed to find bufferization.to_memref for load operation");
  }

  // Map store operations to private memrefs (starting after loads)
  // Trace: store data -> bufferization.to_tensor -> memref.alloc <- subview <-
  // linalg output
  for (size_t i = 0; i < store_ops_.size(); ++i) {
    auto data_tile = store_ops_[i].getDataTile();

    // data_tile should be a bufferization.to_tensor result
    auto to_tensor = data_tile.getDefiningOp<bufferization::ToTensorOp>();
    if (!to_tensor) {
      store_ops_[i].emitError(
          "Store data operand is not a bufferization.to_tensor result");
      signalPassFailure();
      return;
    }

    // Get the memref operand of to_tensor (this is the memref.alloc)
    Value alloc_memref = to_tensor.getOperand();

    // Map this alloc to the private memref
    value_map[alloc_memref] = private_op.getResult(load_ops_.size() + i);
    LLVM_DEBUG(llvm::dbgs()
               << "  Mapped store " << i << " (alloc memref) to private#"
               << (load_ops_.size() + i) << "\n");
  }

  // Clone each linalg operation with remapped operands
  for (auto compute_op : compute_ops_) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Cloning compute op: " << compute_op->getName() << "\n");

    SmallVector<Value> new_operands;

    for (size_t i = 0; i < compute_op->getNumOperands(); ++i) {
      auto operand = compute_op->getOperand(i);

      // Check if operand is a subview
      auto subview = operand.getDefiningOp<memref::SubViewOp>();
      assert(subview && "Expected all linalg operands to be subviews");

      // Get the source of the subview
      Value source = subview.getSource();

      // Must find source in value map
      auto it = value_map.find(source);
      assert(it != value_map.end() &&
             "Failed to map subview source to private memref");

      new_operands.push_back(it->second);
      LLVM_DEBUG(llvm::dbgs() << "    Remapped subview to private memref\n");
    }

    // Clone the operation with new operands and regions
    OperationState state(loc, compute_op->getName());
    state.addOperands(new_operands);
    state.addAttributes(compute_op->getAttrs());
    for (auto result_type : compute_op->getResultTypes()) {
      state.addTypes(result_type);
    }

    // Add regions from the original operation
    for (Region& region : compute_op->getRegions()) {
      Region* new_region = state.addRegion();
      IRMapping mapping;
      region.cloneInto(new_region, mapping);
    }

    builder.create(state);
    LLVM_DEBUG(llvm::dbgs() << "    Created cloned operation with regions\n");
  }
}

void ConstructThreeStagePipelinePass::replaceAccessTilesWithReinterpretCast(
    ModuleOp module) {
  llvm::dbgs() << "BEFORE module " << module << "\n\n";
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
