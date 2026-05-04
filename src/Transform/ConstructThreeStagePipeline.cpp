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

  // Run one-shot bufferization on the module
  LogicalResult bufferizeModule(ModuleOp module);

  // Determine tile sizes based on STICK_SIZE from linalg operation
  SmallVector<int64_t> determineTileSizes(linalg::LinalgOp linalgOp);

  // Transform linalg operations to loops
  void lowerLinalgToLoops(ModuleOp module);

  // Create ktdf.pipeline with ktdf.private and ktdf.stage operations
  void createPipeline(scf::ForOp innermost_loop);
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
  const int64_t STICK_SIZE = schedulerExtContext().getStickSize();
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

  // Start from rightmost dimension and multiply until we reach STICK_SIZE
  int64_t product = 1;
  int64_t coveredDims = 0;

  for (int64_t i = rank - 1; i >= 0; --i) {
    product *= shape[i];
    coveredDims++;
    if (product >= STICK_SIZE) {
      break;
    }
  }

  // Build tile sizes: 1 for uncovered dims, partial/full for covered dims
  tileSizes.resize(rank);
  for (int64_t i = 0; i < rank - coveredDims; ++i) {
    tileSizes[i] = 1;
  }

  // For covered dimensions, compute tile sizes that divide evenly
  int64_t remainingProduct = STICK_SIZE;
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
  module.walk([&](linalg::LinalgOp op) { linalg_ops.push_back(op); });

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

    // Find the innermost loop and create pipeline
    if (!tiled_result->loops.empty()) {
      auto innermost_loop = llvm::cast<scf::ForOp>(tiled_result->loops.back());
      createPipeline(innermost_loop);
    }
  }
}

void ConstructThreeStagePipelinePass::createPipeline(
    scf::ForOp innermost_loop) {
  LLVM_DEBUG(llvm::dbgs() << "Creating ktdf.pipeline with three stages\n");

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

  // Create ktdf.private operation that yields three tokens
  auto token_type = mlir::ktdf::TokenType::get(&getContext());
  SmallVector<Type> private_result_types = {token_type, token_type, token_type};

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

  // Yield the three tokens
  mlir::ktdf::PrivateYieldOp::create(
      builder, innermost_loop.getLoc(),
      ValueRange{t1.getResult(), t2.getResult(), t3.getResult()});

  // Back to pipeline body to create stages
  builder.setInsertionPointToEnd(pipeline_body);

  // Create three stages with sequential dependencies
  mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{},
      /*depends_out=*/ValueRange{private_op.getResult(0)});

  mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{private_op.getResult(0)},
      /*depends_out=*/ValueRange{private_op.getResult(1)});

  mlir::ktdf::StageOp::create(
      builder, innermost_loop.getLoc(),
      /*depends_in=*/ValueRange{private_op.getResult(1)},
      /*depends_out=*/ValueRange{private_op.getResult(2)});
}

void ConstructThreeStagePipelinePass::runOnOperation() {
  if (DisableThisPass) return;

  ModuleOp module = getOperation();

  LLVM_DEBUG(
      llvm::dbgs() << "Starting ConstructThreeStagePipeline transformation\n");

  // Step 1: Bufferize the module (convert tensors to memrefs)
  if (failed(bufferizeModule(module))) {
    module.emitError("Failed to bufferize module");
    signalPassFailure();
    return;
  }
  llvm::dbgs() << module << "\n";
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
