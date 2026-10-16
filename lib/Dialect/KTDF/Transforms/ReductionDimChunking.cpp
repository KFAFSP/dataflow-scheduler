//===-- ReductionDimChunking.cpp --------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// ReductionDimChunking: split reduction dimensions of the inner linalg.generic
// into sequential chunks so that each chunk fits in the hardware FIFO path.
//
// When the chunk sizes are inferred from the threshold (--num-chunks not
// given), ReductionFootprintAnalysis selects them: the reduction dimensions are
// shrunk from the outermost to the innermost in the layout of the inputs until
// the input tiles of a chunk fit the threshold.  The innermost overall loop
// dimension is never chunked.
//
// When --num-chunks is given explicitly, multi-dim chunking is applied using
// one value per reduction dimension (in iterator-type order); one nested
// scf.for is emitted per dim; dims with chunk count 1 produce no loop.
//
// The pass operates on the shape produced by StageCoarsening.  It expects a
// top-level ktdf.pipeline with three sibling stages:
//
//   stage-0 (Load)   : global memory → local memory
//   stage-1 (Compute): outer scf.for (batch) whose body is a single inner
//                       ktdf.pipeline with three stages:
//                         stage-a (Load)   : local memory → FIFO
//                         stage-b (Compute): linalg.generic reduce
//                         stage-c (Store)  : FIFO → local memory
//   stage-2 (Store)  : local memory → global memory
//
// The transformation wraps the single inner ktdf.pipeline in nested scf.for
// loops — one per reduction dimension that has more than one chunk — and
// rewrites it in place to process one chunk per innermost iteration.
// Dimensions whose num_chunks==1 produce no loop (the single chunk covers the
// entire dimension).  The compute stage computes its chunk with the op tiling
// with full-reduction semantics yields.  First-vs-rest accumulation behaviour
// is selected at runtime via
// %condition = AND(iv_j == 0 for all active chunk loop IVs):
//
//   Load stage   : transfers the current input chunk slice (memref → fifo_in)
//                  instead of the whole input.
//                  When !condition, also transfers the partial accumulator
//                  (local memory output buffer → fifo_partial) so the
//                  Compute stage can read it back.
//   Compute stage: unconditionally reads fifo_in and runs the tiled
//                  linalg.generic into the original init.  Then branches on
//                  condition:
//                    then (first iter): yield first generic result.
//                    else (later iters): read fifo_partial, run a second
//                      linalg.generic(input=partial, output=first result),
//                      yield second generic result.
//                  The conditional result is written to fifo_out.
//   Store stage  : unchanged: unconditionally writes fifo_out back to the
//                  local memory output buffer.
//
// The existing local memory output buffer (discovered via the original Store
// stage's data_transfer destination) is reused as the partial-accumulation
// buffer; no new memref is allocated.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>
#include <mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Interfaces/TilingInterface.h>

#include <functional>
#include <memory>
#include <tuple>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionFootprint.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"

#define PASS_NAME "reduction-dim-chunking"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;
using namespace mlir::ktdf;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONDIMCHUNKINGPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Reduction Dim Chunking pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// Helper: find the unique linalg.generic with a reduction iterator inside a
// ktdf.stage body.  Returns nullptr if not found.
// ---------------------------------------------------------------------------
static linalg::GenericOp findReductionGenericOp(ktdf::StageOp stage) {
  linalg::GenericOp found;
  stage.getBody()->walk([&](linalg::GenericOp generic) {
    for (auto it : generic.getIteratorTypesArray()) {
      if (it == utils::IteratorType::reduction) {
        found = generic;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

// ---------------------------------------------------------------------------
// Pass implementation
// ---------------------------------------------------------------------------
struct ReductionDimChunkingPass
    : public ktdf::impl::ReductionDimChunkingPassBase<
          ReductionDimChunkingPass> {
  using ReductionDimChunkingPassBase<
      ReductionDimChunkingPass>::ReductionDimChunkingPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    ReductionDimChunkingPassBase::getDependentDialects(registry);
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    ModuleOp module = getOperation();
    if (failed(transformModule(module))) signalPassFailure();
  }

 private:
  // -----------------------------------------------------------------------
  // Walk the module for linalg.generic ops that have at least one reduction
  // iterator.  For each such op, locate the immediately enclosing ktdf.stage
  // (the Compute stage) and collect unique stages into a list.  The list is
  // processed after the walk to avoid mutating IR while walking.
  // -----------------------------------------------------------------------
  LogicalResult transformModule(ModuleOp module) {
    // Use a DenseSet to avoid O(n²) duplicate checks.
    llvm::DenseSet<ktdf::StageOp> seen;
    SmallVector<ktdf::StageOp> compute_stages;
    module.walk([&](linalg::GenericOp generic) {
      // Check if this generic has a reduction iterator.
      bool has_reduction = false;
      for (auto it : generic.getIteratorTypesArray()) {
        if (it == utils::IteratorType::reduction) {
          has_reduction = true;
          break;
        }
      }
      if (!has_reduction) return WalkResult::advance();

      // linalg.generic is always directly nested inside a ktdf.stage
      // (structural invariant enforced by StageCoarsening).
      auto stage = generic->getParentOfType<ktdf::StageOp>();
      assert(stage && "linalg.generic not inside a ktdf.stage");

      if (seen.insert(stage).second) compute_stages.push_back(stage);
      return WalkResult::advance();
    });

    for (auto compute_stage : compute_stages)
      if (failed(transformPipeline(compute_stage))) return failure();

    return success();
  }

  // -----------------------------------------------------------------------
  // Entry point for one Compute stage.  Determines the reduction dims, chunk
  // counts, and per-dim chunk sizes, then delegates to rewriteComputeStage().
  // -----------------------------------------------------------------------
  LogicalResult transformPipeline(ktdf::StageOp compute_stage) {
    // Stages are always direct children of a pipeline.
    auto inner_pipeline = cast<ktdf::PipelineOp>(compute_stage->getParentOp());

    // Verify there is exactly one inner pipeline inside the parent region.
    unsigned inner_pipeline_count = 0;
    inner_pipeline->getParentRegion()->walk<WalkOrder::PreOrder>(
        [&](ktdf::PipelineOp) {
          ++inner_pipeline_count;
          return WalkResult::skip();
        });
    assert(inner_pipeline_count == 1 &&
           "expected exactly 1 inner pipeline in parent region");

    ktdf::StageOp load_stage = findLoadStage(inner_pipeline, compute_stage);
    assert(load_stage && "could not find load stage upstream of compute stage");

    ktdf::StageOp store_stage = findStoreStage(inner_pipeline, compute_stage);
    assert(store_stage &&
           "could not find store stage downstream of compute stage");

    // Locate the linalg.generic with a reduction iterator.
    linalg::GenericOp generic_op = findReductionGenericOp(compute_stage);
    assert(generic_op && "no reduction linalg.generic in compute stage");

    // FIXME: Current implementation leads to invalid IR when processing
    //        contractions, so skip them for now to allow matmul to pass.
    if (linalg::isaContractionOpInterface(generic_op)) {
      LDBG() << "ignoring contraction";
      return success();
    }

    // ------------------------------------------------------------------
    // Determine reduction dims and per-dim chunk sizes.
    //
    // Threshold path (numChunks empty): ReductionFootprintAnalysis selects the
    // chunk sizes; the innermost overall loop dim is excluded if it is a
    // reduction dim.
    //
    // Explicit path (numChunks non-empty): one value per reduction dim;
    // nested scf.for loops; dims with count 1 produce no loop.
    // ------------------------------------------------------------------

    // FIXME: This implementation assumes that there is an injective mapping
    //        from loop to shape space, i.e., for each iteration, there will be
    //        a distinct element read from the first input tensor.
    SmallVector<unsigned> reduction_dims;
    generic_op.getReductionDims(reduction_dims);
    SmallVector<int64_t> red_dim_sizes;
    SmallVector<int64_t> chunk_sizes;
    const auto loop_ranges = generic_op.getStaticLoopRanges();

    if (numChunks.empty()) {
      // Collect all reduction dims, excluding the innermost overall loop dim
      // if it is a reduction (the innermost loop dim must never be chunked,
      // regardless of iterator type).
      if (!reduction_dims.empty() &&
          reduction_dims.back() == generic_op.getNumLoops() - 1) {
        reduction_dims.pop_back();
      }
      if (reduction_dims.empty()) {
        LDBG(1) << PASS_NAME
            ": could not find a chunkable reduction dimension — skipping";
        return success();
      }

      const ReductionFootprintAnalysis footprint(generic_op);
      if (!footprint.isValid()) {
        inner_pipeline.emitError(PASS_NAME
                                 ": ReductionFootprintAnalysis could not model "
                                 "the reduction");
        return failure();
      }
      const auto tile_sizes = footprint.getTileSizes(chunkSizeThreshold);
      if (!tile_sizes) {
        inner_pipeline.emitError(llvm::Twine(PASS_NAME
                                             ": no chunking of the reduction "
                                             "fits the threshold of ") +
                                 llvm::Twine(chunkSizeThreshold) + " bytes");
        return failure();
      }

      for (const auto dim : reduction_dims) {
        red_dim_sizes.push_back(loop_ranges[dim]);
        chunk_sizes.push_back((*tile_sizes)[dim]);
      }
    } else {
      // User-supplied --num-chunks path: collect reduction dims and validate.
      if (reduction_dims.empty()) {
        inner_pipeline.emitError(PASS_NAME
                                 ": could not find reduction dimension");
        return failure();
      }

      if (numChunks.size() != reduction_dims.size()) {
        inner_pipeline.emitError(
            llvm::Twine(PASS_NAME ": numChunks has ") +
            llvm::Twine(numChunks.size()) + " entries but there are " +
            llvm::Twine(reduction_dims.size()) + " reduction dims");
        return failure();
      }

      for (size_t j = 0; j < reduction_dims.size(); ++j) {
        const auto dim = reduction_dims[j];
        const auto dim_size = loop_ranges[dim];
        red_dim_sizes.push_back(dim_size);
        if (dim_size == ShapedType::kDynamic) {
          LDBG(1) << PASS_NAME
                  << ": dynamic reduction size not yet supported — skipping";
          return success();
        }
        unsigned nchunk = numChunks[j];
        if (nchunk == 0 || dim_size % nchunk != 0) {
          LDBG(1) << PASS_NAME ": num_chunks=" << nchunk
                  << " does not evenly divide dim " << dim
                  << " size=" << dim_size << " — skipping";
          return success();
        }
        chunk_sizes.push_back(dim_size / static_cast<int64_t>(nchunk));
      }
    }

    // Collect per-dim chunk counts in reduction-dim order.
    SmallVector<int64_t> per_dim_num_chunks;
    for (auto [dim_size, chunk_size] :
         llvm::zip_equal(red_dim_sizes, chunk_sizes)) {
      per_dim_num_chunks.push_back(dim_size / chunk_size);
    }

    LDBG(1) << PASS_NAME ": num_reduction_dims=" << reduction_dims.size();

    // One chunk on every dimension leaves the reduction as it is: the rewrite
    // below would rebuild the pipeline to emit the same program, differing only
    // in how the indices print. It also keeps the pass off a single-level
    // pipeline, which has no batch loop for rewriteComputeStage to index
    // against.
    if (llvm::all_of(per_dim_num_chunks,
                     [](int64_t chunks) { return chunks == 1; })) {
      LDBG(1) << PASS_NAME ": every reduction dim has one chunk, nothing to do";
      return success();
    }

    return rewriteComputeStage(inner_pipeline, load_stage, compute_stage,
                               store_stage, generic_op, reduction_dims,
                               chunk_sizes, per_dim_num_chunks);
  }

  // -----------------------------------------------------------------------
  // Wrap inner_pipeline in nested scf.for loops — one per reduction dimension
  // whose num_chunks > 1 — and rewrite it in place to process one chunk per
  // iteration. Dimensions with num_chunks == 1 need no loop; their IV is
  // treated as the constant 0 for condition computation.
  //
  // For N dims with num_chunks[j] > 1 the emitted structure is:
  //
  //   scf.for %iv_0 = 0 to num_chunks[0] step 1 {
  //     scf.for %iv_1 = 0 to num_chunks[1] step 1 {
  //       ...
  //         %condition = (iv_0 == 0 && iv_1 == 0 && ...)
  //         ktdf.pipeline {        // inner_pipeline
  //           ktdf.private { ... } // fifo_in shrunk to a chunk, + fifo_partial
  //           ktdf.stage { ... }   // Load: a chunk, + the partial
  //           ktdf.stage { ... }   // Compute: the tiled op, + the combine
  //           ktdf.stage { ... }   // Store: unchanged
  //         }
  //     }
  //   }
  //
  // The compute stage computes its chunk with the op tiling yields. Carrying
  // the change of its input shape over to fifo_in and the load stage is done
  // here, by hand.
  // -----------------------------------------------------------------------
  LogicalResult rewriteComputeStage(
      ktdf::PipelineOp inner_pipeline, ktdf::StageOp load_stage,
      ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
      linalg::GenericOp generic_op, ArrayRef<unsigned> reduction_dims,
      ArrayRef<int64_t> chunk_sizes, ArrayRef<int64_t> per_dim_num_chunks) {
    MLIRContext* context = inner_pipeline.getContext();
    IRRewriter rewriter(context);
    Location loc = inner_pipeline.getLoc();

    // ------------------------------------------------------------------
    // Step 1: Trace dataflow to discover the source/destination memrefs and
    // the original FIFO slot types.
    //
    // Input path:
    //   input_memref → DataTransferOp → fifo_in → ReadFromFifoOp
    //               → generic.ins[0]
    //
    // Output path:
    //   generic.result(0) → WriteToFifoOp → fifo_out
    //               → DataTransferOp → partial_memref
    // ------------------------------------------------------------------
    ktdf::DataTransferOp load_transfer;
    ktdf::ReadFromFifoOp read_from_fifo;
    std::tie(load_transfer, read_from_fifo) =
        findLoadTransfer(load_stage, generic_op);
    if (!read_from_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic input is not produced by read_from_fifo");
      return failure();
    }
    if (!load_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no memref-to-fifo transfer feeding "
                               "the generic's fifo_in slot in load stage");
      return failure();
    }
    if (!load_transfer.getSourceSizes().empty() ||
        !load_transfer.getStaticSourceSizes()) {
      inner_pipeline.emitError(
          PASS_NAME ": expected the load transfer to have static sizes");
      return failure();
    }

    ktdf::DataTransferOp store_transfer;
    ktdf::WriteToFifoOp write_to_fifo;
    std::tie(store_transfer, write_to_fifo) =
        findStoreTransfer(compute_stage, store_stage, generic_op);
    if (!write_to_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic result is not consumed by write_to_fifo");
      return failure();
    }
    if (!store_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no fifo-to-memref transfer consuming "
                               "the generic's fifo_out slot in store stage");
      return failure();
    }
    if (!store_transfer.getDestSizes().empty() ||
        !store_transfer.getStaticDestSizes()) {
      inner_pipeline.emitError(
          PASS_NAME ": expected the store transfer to have static sizes");
      return failure();
    }

    auto in_fifo_type =
        dyn_cast<ktdf::FifoSlotType>(read_from_fifo.getFifoSlot().getType());
    auto out_fifo_type =
        dyn_cast<ktdf::FifoSlotType>(write_to_fifo.getFifoSlot().getType());
    if (!in_fifo_type || !out_fifo_type) {
      inner_pipeline.emitError(
          PASS_NAME
          ": could not derive input/output FIFO slot types "
          "from compute stage");
      return failure();
    }

    // The batch loop is a precondition: without it, the stages loading into
    // local memory share the pipeline, and would be repeated for every chunk.
    // A single-level pipeline is well-formed IR, so report the missing loop
    // rather than assert on it.
    if (!isa<scf::ForOp>(inner_pipeline->getParentRegion()->getParentOp()))
      return inner_pipeline.emitError(PASS_NAME)
             << ": reduction needs chunking, but its pipeline is not nested in "
                "a batch scf.for, so there is no index to chunk against";

    // ------------------------------------------------------------------
    // Step 2: Tile the reduction with full-reduction semantics along every
    // chunked dim, leaving the others whole.
    // ------------------------------------------------------------------
    SmallVector<int64_t> tile_sizes(generic_op.getNumLoops(), 0);
    for (auto [dim, chunk_size, num_chunks] :
         llvm::zip_equal(reduction_dims, chunk_sizes, per_dim_num_chunks)) {
      if (num_chunks != 1) tile_sizes[dim] = chunk_size;
    }
    scf::SCFTilingOptions tiling_options;
    tiling_options.setLoopType(scf::SCFTilingOptions::LoopType::ForOp)
        .setTileSizes(getAsIndexOpFoldResult(context, tile_sizes))
        .setReductionTilingStrategy(ReductionTilingStrategy::FullReduction);
    rewriter.setInsertionPoint(generic_op);
    auto tiling = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(generic_op.getOperation()),
        tiling_options);
    if (failed(tiling)) {
      inner_pipeline.emitError(PASS_NAME ": tiling of the reduction failed");
      return failure();
    }
    auto chunk_op = cast<linalg::GenericOp>(tiling->tiledOps.front());
    auto slice = chunk_op.getDpsInputOperand(0)
                     ->get()
                     .getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice) {
      inner_pipeline.emitError(
          PASS_NAME ": expected the tiled reduction to read a slice");
      return failure();
    }
    auto chunk_input_tensor_type = slice.getResultType();

    // The tiled loop each dim of the slice is offset by, if any, and the size
    // of a step of each.
    SmallVector<std::optional<unsigned>> offset_loops;
    for (OpFoldResult offset : slice.getMixedOffsets()) {
      if (isConstantIntValue(offset, 0)) {
        offset_loops.push_back(std::nullopt);
        continue;
      }
      const auto* const it =
          llvm::find_if(tiling->loops, [&](LoopLikeOpInterface loop) {
            return loop.getSingleInductionVar() ==
                   dyn_cast_if_present<Value>(offset);
          });
      if (it == tiling->loops.end()) {
        inner_pipeline.emitError(PASS_NAME
                                 ": expected the tiled reduction to slice its "
                                 "input at the tiled loop IVs");
        return failure();
      }
      offset_loops.push_back(static_cast<unsigned>(it - tiling->loops.begin()));
    }
    SmallVector<int64_t> tile_steps;
    for (auto size : tile_sizes) {
      if (size != 0) tile_steps.push_back(size);
    }

    // ------------------------------------------------------------------
    // Step 3: Wrap inner_pipeline in the chunk loops.
    //
    // Materialise per-dim upper-bound constants before the outermost loop.
    // These are index-typed scf.for bounds and must live at the scf.for
    // scope, not inside any stage body.
    // ------------------------------------------------------------------
    rewriter.setInsertionPoint(inner_pipeline);
    size_t n_dims = reduction_dims.size();
    SmallVector<Value> c_per_dim_num_chunks;
    for (int64_t nc : per_dim_num_chunks) {
      c_per_dim_num_chunks.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, nc));
    }

    // Build loops from outermost (dim 0) to innermost (dim n_dims-1), one for
    // each dim with num_chunks > 1. chunk_ivs holds their IVs, in the order of
    // the tiled loops.
    SmallVector<Value> chunk_ivs;
    Value c0_loop = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1_loop = arith::ConstantIndexOp::create(rewriter, loc, 1);
    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) continue;
      auto loop = scf::ForOp::create(rewriter, loc, c0_loop,
                                     c_per_dim_num_chunks[j], c1_loop);
      chunk_ivs.push_back(loop.getInductionVar());
      // Descend into the loop body for the next (inner) level.
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // condition = AND of (iv_j == 0) for every dim whose loop was generated.
    Value condition;
    for (Value iv : chunk_ivs) {
      Value eq = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                       iv, c0_loop);
      condition =
          condition ? arith::AndIOp::create(rewriter, loc, condition, eq) : eq;
    }
    rewriter.moveOpBefore(inner_pipeline, rewriter.getInsertionBlock(),
                          rewriter.getInsertionPoint());

    // ------------------------------------------------------------------
    // Step 4: Add fifo_partial, which carries output-tensor-sized elements in
    // the Load → Compute direction (same src/dest endpoints as fifo_in),
    // allocated right after fifo_in, and shrink fifo_in to a chunk. The
    // PrivateOp is rebuilt once fifo_partial is in use.
    // ------------------------------------------------------------------
    ktdf::PrivateBuilder slots(rewriter, inner_pipeline, loc);
    auto fifo_in = cast<OpResult>(read_from_fifo.getFifoSlot());
    auto fifo_in_alloc = cast<ktdf::PrivateOp>(fifo_in.getOwner())
                             .getYieldOp()
                             ->getOperand(fifo_in.getResultNumber());
    Value fifo_partial =
        slots
            .createFifo(ktdf::FifoSlotType::get(
                context, in_fifo_type.getSrc(), in_fifo_type.getDest(),
                out_fifo_type.getNumElements(), out_fifo_type.getElementType()))
            .front();
    rewriter.moveOpAfter(fifo_partial.getDefiningOp(),
                         fifo_in_alloc.getDefiningOp());
    const auto chunk_elements = chunk_input_tensor_type.getNumElements();
    rewriter.modifyOpInPlace(fifo_in_alloc.getDefiningOp(), [&]() {
      fifo_in_alloc.setType(ktdf::FifoSlotType::get(
          context, in_fifo_type.getSrc(), in_fifo_type.getDest(),
          chunk_elements, in_fifo_type.getElementType()));
    });

    // ------------------------------------------------------------------
    // Step 5: The compute stage reads a chunk into the tiled op, and combines
    // its result with the running partial on non-first iterations:
    //
    //   then: yield chunk_result                       (no prior partial)
    //   else: partial ← read_from_fifo(fifo_partial)
    //         combined ← linalg.generic(partial, chunk_result)
    //         yield combined
    // ------------------------------------------------------------------
    Value init = generic_op.getDpsInits().front();
    rewriter.moveOpBefore(chunk_op, tiling->loops.front());
    rewriter.modifyOpInPlace(read_from_fifo, [&]() {
      read_from_fifo.getResult().setType(chunk_input_tensor_type);
    });
    rewriter.modifyOpInPlace(chunk_op, [&]() {
      chunk_op.getDpsInputOperand(0)->set(read_from_fifo);
      chunk_op.getDpsInitOperand(0)->set(init);
    });
    eraseTiledLoops(rewriter, *tiling);

    rewriter.setInsertionPointAfter(chunk_op);
    auto output_tensor_type = cast<RankedTensorType>(init.getType());
    auto if_op = scf::IfOp::create(rewriter, loc, TypeRange{output_tensor_type},
                                   condition, /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(if_op.thenBlock());
    scf::YieldOp::create(rewriter, loc, chunk_op->getResult(0));
    rewriter.setInsertionPointToStart(if_op.elseBlock());
    Value partial = ktdf::ReadFromFifoOp::create(
        rewriter, loc, output_tensor_type, fifo_partial);
    scf::YieldOp::create(rewriter, loc,
                         createCombine(rewriter, loc, chunk_op, partial,
                                       chunk_op->getResult(0)));
    rewriter.replaceAllUsesWith(generic_op->getResult(0), if_op.getResult(0));
    rewriter.eraseOp(generic_op);

    // ------------------------------------------------------------------
    // Step 6: The load stage transfers a chunk, offset by the chunk IVs, and
    // on non-first iterations the partial accumulator from where the store
    // stage stores it.
    //
    // Input memref layout: dim-0 is the batch/local slot index; dims
    // 1..rank-1 correspond to the dims of the generic's input. The offset of a
    // chunked dim is encoded as iv * chunk_size in the AffineMap (to avoid
    // emitting arith.muli, which the backend does not accept as an index).
    // ------------------------------------------------------------------
    const auto source_type =
        cast<MemRefType>(load_transfer.getSource().getType());
    if (source_type.getRank() != chunk_input_tensor_type.getRank() + 1) {
      inner_pipeline.emitError(PASS_NAME
                               ": expected the input memref to have a batch "
                               "dim ahead of the dims of the generic's input");
      return failure();
    }
    const auto num_indices =
        static_cast<unsigned>(load_transfer.getSourceIndices().size());
    const auto source_map = load_transfer.getSourceMap().value_or(
        AffineMap::getMultiDimIdentityMap(num_indices, context));
    SmallVector<AffineExpr> source_exprs(source_map.getResults());
    auto source_sizes = llvm::to_vector(*load_transfer.getStaticSourceSizes());
    for (auto [dim, loop, size] :
         llvm::enumerate(offset_loops, chunk_input_tensor_type.getShape())) {
      if (loop) {
        source_exprs[dim + 1] =
            source_exprs[dim + 1] +
            getAffineDimExpr(num_indices + *loop, context) * tile_steps[*loop];
      }
      source_sizes[dim + 1] = size;
    }
    rewriter.modifyOpInPlace(load_transfer, [&]() {
      load_transfer.setSourceMapAttr(AffineMapAttr::get(
          AffineMap::get(num_indices + static_cast<unsigned>(chunk_ivs.size()),
                         0, source_exprs, context)));
      load_transfer.getSourceIndicesMutable().append(chunk_ivs);
      load_transfer.setStaticSourceSizes(source_sizes);
      load_transfer.setStaticDestSizes({chunk_elements});
    });

    rewriter.setInsertionPointAfter(load_transfer);
    auto partial_indices =
        cloneOutOf(rewriter, store_transfer.getDestIndices(), store_stage);
    auto load_partial =
        scf::IfOp::create(rewriter, loc, /*resultTypes=*/TypeRange{}, condition,
                          /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(load_partial.elseBlock());
    ktdf::DataTransferOp::create(
        rewriter, loc, store_transfer.getDestination(),
        store_transfer.getDestMap().value_or(AffineMap{}), partial_indices,
        *store_transfer.getStaticDestSizes(), fifo_partial, AffineMap{},
        ValueRange{}, ArrayRef<int64_t>{out_fifo_type.getNumElements()});
    DominanceInfo dominance;
    eliminateCommonSubExpressions(rewriter, dominance, load_stage);

    return success();
  }

  // Erases the loops tiling left around the tiled op, which was moved out of
  // them, along with what tiling created only for them.
  static void eraseTiledLoops(RewriterBase& rewriter,
                              const scf::SCFTilingResult& tiling) {
    Operation* outermost = tiling.loops.front();
    llvm::SetVector<Operation*> bounds;
    for (LoopLikeOpInterface loop : tiling.loops) {
      for (Value operand : loop->getOperands()) {
        Operation* def = operand.getDefiningOp();
        if (def && !outermost->isAncestor(def)) bounds.insert(def);
      }
    }
    rewriter.eraseOp(outermost);
    for (Operation* def : bounds) {
      if (isOpTriviallyDead(def)) rewriter.eraseOp(def);
    }
  }

  // Returns `values`, cloning the pure ops computing them within `scope` to
  // the insertion point of `rewriter`.
  static SmallVector<Value> cloneOutOf(RewriterBase& rewriter,
                                       ValueRange values, Operation* scope) {
    IRMapping mapping;
    std::function<Value(Value)> get = [&](Value value) -> Value {
      if (Value mapped = mapping.lookupOrNull(value)) return mapped;
      Operation* def = value.getDefiningOp();
      if (!def || !scope->isProperAncestor(def)) return value;
      assert(isPure(def) && def->getNumRegions() == 0 &&
             "index computations are expected to be pure");
      for (Value operand : def->getOperands()) (void)get(operand);
      rewriter.clone(*def, mapping);
      return mapping.lookup(value);
    };
    return llvm::map_to_vector(values, get);
  }

  // Creates a linalg.generic combining `partial` into `chunk_result`, with the
  // body of `chunk_op`. Both are of the output tensor type, so it uses
  // identity maps for both operands and all-parallel iterators — the chunk op
  // has a higher-rank input map and cannot be cloned directly here.
  static Value createCombine(RewriterBase& rewriter, Location loc,
                             linalg::GenericOp chunk_op, Value partial,
                             Value chunk_result) {
    auto output_tensor_type = cast<RankedTensorType>(chunk_result.getType());
    int64_t out_rank = output_tensor_type.getRank();
    AffineMap id_map =
        AffineMap::getMultiDimIdentityMap(out_rank, rewriter.getContext());
    SmallVector<utils::IteratorType> parallel_iters(
        static_cast<size_t>(out_rank), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexing_maps = {id_map, id_map};
    auto combine_generic = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/TypeRange{output_tensor_type},
        /*inputs=*/ValueRange{partial},
        /*outputs=*/ValueRange{chunk_result},
        /*indexingMaps=*/indexing_maps,
        /*iteratorTypes=*/parallel_iters);
    Type elem_type = output_tensor_type.getElementType();
    Block* new_body = &combine_generic.getRegion().emplaceBlock();
    new_body->addArgument(elem_type, loc);  // ins[0] element
    new_body->addArgument(elem_type, loc);  // outs[0] element

    // Clone the scalar body ops from the chunk op, remapping its block args to
    // the new block args.
    IRMapping body_map;
    Block* orig_body = chunk_op.getBody();
    body_map.map(orig_body->getArgument(0), new_body->getArgument(0));
    body_map.map(orig_body->getArgument(1), new_body->getArgument(1));
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(new_body);
    for (auto& op : *orig_body) rewriter.clone(op, body_map);
    return combine_generic.getResult(0);
  }
};

}  // namespace

auto mlir::ktdf::createReductionDimChunkingPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>();
}
