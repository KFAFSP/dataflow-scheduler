//===-- MapReductionPartials.cpp --------------------------------*- c++ -*-===//
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
// MapReductionPartials: lower reduction linalg.generic ops whose first input
// comes from a ktdf.read_from_fifo to buffer-semantics form backed by a
// pre-allocated accumulator memref.
//
// After ReductionLoopExposure the compute stage looks like:
//
//   %empty = tensor.empty() : tensor<CxDxf16>
//   %result = scf.for %r = 0 to R iter_args(%carry = %empty)
//                 {loop_type = reduction_loop}
//     %slice = ktdf.read_from_fifo %fifo -> tensor<CxDxf16>
//     %new_carry = linalg.generic ins(%slice) outs(%carry)
//     scf.if (%r == R-1): ktdf.write_to_fifo %new_carry, %fifo_out
//     scf.yield %new_carry
//
// This pass rewrites that to:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   linalg.fill(%zero, %alloc)
//   scf.for %r = 0 to R {
//     %slice = ktdf.read_from_fifo %fifo -> memref<CxDxf16>
//     linalg.generic ins(%slice) outs(%alloc)
//     scf.if (%r == R-1): ktdf.write_to_fifo %alloc, %fifo_out
//   }
//
// When the iter_arg initializer is a conditional (scf.if), the pass recurses
// into each branch and handles the yielded value:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   scf.if %cond {                     // no result; was -> (tensor<...>)
//     linalg.fill(%zero, %alloc)       // was: tensor.empty + yield
//   } else {
//     %r = ktdf.read_from_fifo ...     // memref form of read_from_fifo
//     memref.copy %r, %alloc
//   }
//   scf.for %r = 0 to R { ... }       // iter_arg stripped
//
// The iter_arg / loop result / scf.yield operand are stripped, and any
// write_to_fifo that consumed the loop result is patched to use %alloc.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "map-reduction-partials"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Map Reduction Partials pass"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_MAPREDUCTIONPARTIALSPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

// ---------------------------------------------------------------------------
// Return true if `generic_op` has at least one reduction iterator.
// ---------------------------------------------------------------------------
static bool hasReductionIterator(linalg::GenericOp generic_op) {
  for (auto it : generic_op.getIteratorTypesArray())
    if (it == utils::IteratorType::reduction) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Transform the initializer `init_val` of a loop-carried accumulator in-place,
// emitting linalg.fill (or memref.copy) into `alloc_val` wherever the original
// tensor value was produced.  `alloc_val` is a memref.alloc already emitted at
// the top of the stage.
//
// Three leaf cases are handled; scf.if recurses into both branches:
//
//   tensor.empty()
//     → linalg.fill(%zero, %alloc) inserted just before the tensor.empty,
//       then the tensor.empty is erased.
//
//   ktdf.read_from_fifo ... -> tensor<...>
//     → emit memref-typed read_from_fifo + memref.copy into %alloc just
//       before the original op, then erase it.
//
//   scf.if %cond -> (tensor<...>) { yield A } else { yield B }
//     → recurse on A (then-branch yield operand 0) and B (else-branch yield
//       operand 0), drop the yield operands so both yields become result-less,
//       then rebuild the scf.if with no result types.
// ---------------------------------------------------------------------------
static LogicalResult lowerIterArgInitializer(Value init_val, Value alloc_val) {
  Operation* defining_op = init_val.getDefiningOp();
  assert(defining_op &&
         "lowerIterArgInitializer: init_val must be an op result");

  Type elem_type = cast<MemRefType>(alloc_val.getType()).getElementType();

  // ── Case: tensor.empty ────────────────────────────────────────────────────
  if (auto empty_op = dyn_cast<tensor::EmptyOp>(defining_op)) {
    OpBuilder builder(empty_op);
    Location loc = empty_op.getLoc();
    Value zero = arith::ConstantOp::create(
        builder, loc, builder.getFloatAttr(elem_type, 0.0));
    linalg::FillOp::create(builder, loc, ValueRange{zero},
                           ValueRange{alloc_val});
    empty_op->erase();
    return success();
  }

  // ── Case: ktdf.read_from_fifo returning a tensor ──────────────────────────
  if (auto read_op = dyn_cast<ktdf::ReadFromFifoOp>(defining_op)) {
    auto tensor_type = cast<RankedTensorType>(read_op.getResult().getType());
    auto memref_type =
        MemRefType::get(tensor_type.getShape(), tensor_type.getElementType());
    OpBuilder builder(read_op);
    Location loc = read_op.getLoc();
    Value new_read = ktdf::ReadFromFifoOp::create(builder, loc, memref_type,
                                                  read_op.getFifoSlot())
                         .getResult();
    memref::CopyOp::create(builder, loc, new_read, alloc_val);
    read_op->erase();
    return success();
  }

  // ── Case: scf.if returning a tensor — recurse into both branches ──────────
  if (auto if_op = dyn_cast<scf::IfOp>(defining_op)) {
    // Drop the yield operand *before* recursing so that when the recursive
    // call erases the defining op (e.g. tensor.empty) no uses remain.
    {
      auto then_yield =
          cast<scf::YieldOp>(if_op.getThenRegion().front().getTerminator());
      Value then_init = then_yield.getOperand(0);
      then_yield->setOperands({});
      if (failed(lowerIterArgInitializer(then_init, alloc_val)))
        return failure();
    }

    {
      auto else_yield =
          cast<scf::YieldOp>(if_op.getElseRegion().front().getTerminator());
      Value else_init = else_yield.getOperand(0);
      else_yield->setOperands({});
      if (failed(lowerIterArgInitializer(else_init, alloc_val)))
        return failure();
    }

    // Rebuild the scf.if without result types and transfer the rewritten
    // regions.
    OpBuilder builder(if_op);
    auto new_if =
        scf::IfOp::create(builder, if_op.getLoc(),
                          /*resultTypes=*/TypeRange{}, if_op.getCondition(),
                          /*withElseRegion=*/true);
    new_if.getThenRegion().takeBody(if_op.getThenRegion());
    new_if.getElseRegion().takeBody(if_op.getElseRegion());
    if_op.erase();
    return success();
  }

  return defining_op->emitError(
      "lowerIterArgInitializer: unrecognised initializer form — expected "
      "tensor.empty, ktdf.read_from_fifo, or scf.if");
}

// ---------------------------------------------------------------------------
// Walk up the iter_arg chain rooted at `acc_iter_arg` (outs[0] of the
// original linalg.generic, captured before erasure) and rebuild each
// enclosing scf.for without that iter_arg slot, replacing every use of
// the dropped iter_arg and its loop result with `alloc_val`.
//
// After the innermost loop is rebuilt the old loop result may itself be an
// iter_arg of an outer scf.for — we keep climbing until the chain exits a
// scf.for.
//
// The initializer at the top of the chain has already been handled by
// lowerIterArgInitializer before this function is called, so it is NOT
// erased here.
//
// Cannot remove an iter_arg in-place: getInitArgsMutable().erase() only drops
// the operand but leaves the region block argument and loop result intact,
// producing an inconsistent op that fails the verifier.  We clone instead.
// ---------------------------------------------------------------------------
static void removeUnusedIterArgChain(BlockArgument acc_iter_arg,
                                     Value alloc_val) {
  // `current` starts as the innermost iter_arg and advances to the init of
  // that iter_arg at each level.
  Value current = acc_iter_arg;
  while (auto iter_arg = dyn_cast<BlockArgument>(current)) {
    auto for_op = dyn_cast<scf::ForOp>(iter_arg.getOwner()->getParentOp());
    assert(for_op);

    // Block arg 0 of an scf.for is the induction variable; iter_args start
    // at index 1, so subtract 1 to get the iter_arg slot index.
    unsigned iter_idx = iter_arg.getArgNumber() - 1;

    // Capture the init before rebuilding — this is what we advance to next.
    Value iter_init = for_op.getInits()[iter_idx];

    // Replace all uses of the dropped iter_arg and its loop result.
    Value loop_result = for_op.getResult(iter_idx);
    loop_result.replaceAllUsesWith(alloc_val);
    iter_arg.replaceAllUsesWith(alloc_val);

    // Rebuild the loop without the iter_arg at `iter_idx`.
    OpBuilder builder(for_op);
    Location loc = for_op.getLoc();

    SmallVector<Value> new_inits;
    for (auto [i, init] : llvm::enumerate(for_op.getInits()))
      if (i != iter_idx) new_inits.push_back(init);

    auto new_for =
        scf::ForOp::create(builder, loc, for_op.getLowerBound(),
                           for_op.getUpperBound(), for_op.getStep(), new_inits);

    for (auto attr : for_op->getAttrs())
      if (attr.getName() != "operandSegmentSizes")
        new_for->setAttr(attr.getName(), attr.getValue());

    // Map old to new iter args, skipping over iter_idx.
    IRMapping body_map;
    body_map.map(for_op.getInductionVar(), new_for.getInductionVar());
    unsigned new_arg_idx = 0;
    for (auto [i, arg] : llvm::enumerate(for_op.getRegionIterArgs()))
      if (i != iter_idx)
        body_map.map(arg, new_for.getRegionIterArgs()[new_arg_idx++]);

    // Copy old scf.for yield operands skipping over iter_idx.
    auto* old_yield = for_op.getBody()->getTerminator();
    Operation* new_yield = new_for.getBody()->getTerminator();
    OpBuilder body_builder(new_yield);
    for (auto& op : for_op.getBody()->without_terminator())
      body_builder.clone(op, body_map);

    SmallVector<Value> new_yield_operands;
    for (auto [i, operand] : llvm::enumerate(old_yield->getOperands()))
      if (i != iter_idx)
        new_yield_operands.push_back(body_map.lookupOrDefault(operand));
    new_yield->setOperands(new_yield_operands);

    unsigned new_res_idx = 0;
    for (auto [i, res] : llvm::enumerate(for_op.getResults()))
      if (i != iter_idx)
        res.replaceAllUsesWith(new_for.getResult(new_res_idx++));

    // Advance to next parent loop in iter arg chain.
    current = iter_init;
    for_op.erase();
  }
}

// ---------------------------------------------------------------------------
// Emit a new ktdf.read_from_fifo with a memref result type that mirrors the
// tensor-typed input at position 0 of `generic_op`.
//
// Asserts that `generic_op` has exactly one input and that the input is
// defined by a ktdf.read_from_fifo — no other producer is supported.
// ---------------------------------------------------------------------------
static Value convertInputToMemref(OpBuilder& builder,
                                  linalg::GenericOp generic_op) {
  assert(generic_op.getInputs().size() == 1 &&
         "convertInputToMemref: expected exactly one input on linalg.generic");
  Value input = generic_op.getInputs()[0];
  auto orig_read = input.getDefiningOp<ktdf::ReadFromFifoOp>();
  assert(orig_read &&
         "convertInputToMemref: input[0] must be a ktdf.read_from_fifo");

  auto in_tensor_type = cast<RankedTensorType>(input.getType());
  auto in_memref_type = MemRefType::get(in_tensor_type.getShape(),
                                        in_tensor_type.getElementType());
  return ktdf::ReadFromFifoOp::create(builder, generic_op.getLoc(),
                                      in_memref_type, orig_read.getFifoSlot())
      .getResult();
}

// ---------------------------------------------------------------------------
// Lower a single reduction linalg.generic to buffer semantics.
// `generic_op` must have at least one reduction iterator and its input must
// be a ktdf.read_from_fifo.
// ---------------------------------------------------------------------------
static LogicalResult rewriteGeneric(
    linalg::GenericOp generic_op,
    scheduler::arch_view::GroupLocalMemory& group_local_mem) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  // Step 1: decide which memory kind backs the accumulator.
  Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
  if (!mem_space) return failure();

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  auto out_tensor_type =
      cast<RankedTensorType>(generic_op.getDpsInitOperand(0)->get().getType());
  auto alloc_type = MemRefType::get(out_tensor_type.getShape(),
                                    out_tensor_type.getElementType(),
                                    MemRefLayoutAttrInterface{}, mem_space);

  // Step 2: allocate the accumulator memref at the top of the stage.
  builder.setInsertionPointToStart(stage.getBody());
  auto alloc = memref::AllocOp::create(builder, loc, alloc_type);

  // Step 3: capture the iter_arg and walk up to find the outermost initializer
  // before anything is erased.
  assert(generic_op.getOutputs().size() == 1 &&
         "rewriteGeneric: expected exactly one output on linalg.generic");
  auto acc_iter_arg = dyn_cast<BlockArgument>(generic_op.getOutputs()[0]);
  assert(acc_iter_arg &&
         isa<scf::ForOp>(acc_iter_arg.getOwner()->getParentOp()) &&
         "rewriteGeneric: outs[0] must be an iter_arg of an scf.for");

  Value outermost_init;
  {
    Value cursor = acc_iter_arg;
    while (auto ba = dyn_cast<BlockArgument>(cursor)) {
      auto for_op = cast<scf::ForOp>(ba.getOwner()->getParentOp());
      assert(for_op);
      outermost_init = for_op.getInits()[ba.getArgNumber() - 1];
      cursor = outermost_init;
    }
  }
  assert(outermost_init && "could not find iter_arg initializer");

  // Step 4: replace all uses of the outermost initializer (e.g. the scf.for
  // init operand) with alloc_val *before* lowering it, so that when
  // lowerIterArgInitializer erases the defining op no uses remain.
  outermost_init.replaceAllUsesWith(alloc.getResult());

  // Step 4b: lower the initializer — emit linalg.fill (and/or memref.copy) in
  // the right place and clean up tensor ops.
  if (failed(lowerIterArgInitializer(outermost_init, alloc.getResult())))
    return failure();

  // Step 5: emit a new ktdf.read_from_fifo with a memref result type so the
  // buffer-semantics linalg.generic below has a pure-buffer input.
  builder.setInsertionPoint(generic_op);
  Value new_read = convertInputToMemref(builder, generic_op);

  // Step 6: pure-buffer linalg.generic — memref ins + memref outs, no result.
  auto buf_generic = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{},
      /*inputs=*/ValueRange{new_read},
      /*outputs=*/ValueRange{alloc.getResult()},
      generic_op.getIndexingMapsAttr(), generic_op.getIteratorTypesAttr(),
      /*doc=*/StringAttr{},
      /*library_call=*/StringAttr{});
  IRMapping mapping;
  generic_op.getRegion().cloneInto(&buf_generic.getRegion(), mapping);
  // cloneInto prepends an empty placeholder block; drop it, keep the clone.
  Block& placeholder = buf_generic.getRegion().front();
  if (&placeholder != &buf_generic.getRegion().back()) placeholder.erase();

  // Step 7: replace generic result with alloc, patch write_to_fifo users,
  // then erase the generic and its now-dead tensor read_from_fifo input.
  Value generic_result = generic_op.getResult(0);
  for (Operation* user : generic_result.getUsers())
    if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user))
      write_op.getDataMutable().assign(alloc.getResult());
  generic_result.replaceAllUsesWith(alloc.getResult());

  Value orig_input = generic_op.getInputs()[0];
  generic_op.erase();
  if (orig_input.use_empty())
    if (auto* orig_input_op = orig_input.getDefiningOp())
      orig_input_op->erase();

  // Step 8: walk up the iter_arg chain and rebuild each scf.for without it.
  // The initializer has already been transformed; no erasure is needed here.
  removeUnusedIterArgChain(acc_iter_arg, alloc.getResult());
  return success();
}

struct MapReductionPartialsPass
    : public ktdf::impl::MapReductionPartialsPassBase<
          MapReductionPartialsPass> {
  using MapReductionPartialsPassBase<
      MapReductionPartialsPass>::MapReductionPartialsPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module->emitError("Unable to import the device specification.");
      signalPassFailure();
      return;
    }
    auto& group_local_mem =
        device_manager.getOrCreateView<scheduler::arch_view::GroupLocalMemory>(
            *device);

    SmallVector<linalg::GenericOp> candidates;
    module.walk([&](linalg::GenericOp generic_op) {
      if (hasReductionIterator(generic_op)) candidates.push_back(generic_op);
    });

    for (auto generic_op : candidates) {
      if (failed(rewriteGeneric(generic_op, group_local_mem))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

auto mlir::ktdf::createMapReductionPartialsPass() -> std::unique_ptr<Pass> {
  return std::make_unique<MapReductionPartialsPass>();
}
