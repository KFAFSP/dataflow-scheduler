//===-- HoistRegisters.cpp -------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Transforms/Passes.h"

namespace scheduler {
#define GEN_PASS_DEF_HOISTREGISTERSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace mlir;

namespace scheduler {
namespace {

/// Gets the mapped constants \p generic 's body reads, in the order it reads
/// them.
///
/// Found by use rather than by where they are defined: a pattern inserts them
/// in the body, but folding materializes a constant in the entry block of the
/// region it is in, so by now most are already outside. The mapping is what
/// says one belongs in a register rather than being an operand of the
/// arithmetic.
auto getMappedConstants(linalg::GenericOp generic)
    -> SmallVector<arith::ConstantOp> {
  // Ordered, because the order they are found in is the order the operands are
  // in, which is what names each of them a register.
  SetVector<Operation*> seen;
  generic.getBody()->walk([&](Operation* op) {
    for (const auto operand : op->getOperands()) {
      const auto constant = operand.getDefiningOp<arith::ConstantOp>();
      if (!constant) continue;
      if (!ktdf_arch::getProperty<ktdf_arch::MapsToAttr>(constant)) continue;
      seen.insert(constant);
    }
  });

  SmallVector<arith::ConstantOp> result;
  for (auto* op : seen) result.push_back(cast<arith::ConstantOp>(op));
  return result;
}

/// Gets how many lanes of \p element a compute unit of \p op 's device holds.
///
/// The unit's SIMD feature says so, per element type, which is the same number
/// a register of it holds -- so a device with wider registers or another
/// element needs nothing here. Zero when the device does not say, which the
/// caller reports rather than guessing a width the template was not written
/// for.
auto getLaneCount(Operation* op, Type element, AnalysisManager analyses)
    -> int64_t {
  const auto declaration = ktdf_arch::findDeviceDeclarationFor(op);
  if (!declaration) return 0;

  // Through a reference, because the declaration a program carries is an import
  // of the description rather than the description itself -- walking the
  // declaration alone finds an empty body.
  ktdf_arch::DeviceRef device(declaration, analyses);

  int64_t result = 0;
  device->getBodyRegion().walk(
      [&](ktdf_arch::ExecutionUnitOp unit) -> WalkResult {
        const auto simd = ktdf_arch::getFeature<ktdf_arch::feature::SIMD>(
            unit.getOperation());
        if (!simd) return WalkResult::advance();
        const auto lanes = simd.getLanes(element);
        if (lanes == 0) return WalkResult::advance();
        result = lanes;
        return WalkResult::interrupt();
      });
  return result;
}

/// Turns \p constant into a register in front of \p generic.
///
/// The register is a whole one, filled with the value repeated: what a template
/// reads out of it is a vector, not one element. Said as a fill, which is what
/// the lowering below turns into the bitstream and the store that write it.
/// Returns the register, which the body then reads instead of the constant.
auto materializeRegister(arith::ConstantOp constant, linalg::GenericOp generic,
                         AnalysisManager analyses, RewriterBase& rewriter)
    -> Value {
  const auto maps_to = ktdf_arch::getProperty<ktdf_arch::MapsToAttr>(constant);
  const auto element = constant.getType();
  const auto lanes = getLaneCount(constant, element, analyses);
  if (lanes == 0) {
    constant.emitError("the device says no lane count for ") << element;
    return nullptr;
  }

  const auto register_type =
      MemRefType::get({lanes}, element, MemRefLayoutAttrInterface{}, maps_to);

  rewriter.setInsertionPoint(generic);
  auto reg =
      memref::AllocOp::create(rewriter, constant.getLoc(), register_type);
  linalg::FillOp::create(rewriter, constant.getLoc(),
                         ValueRange{constant.getResult()}, ValueRange{reg});
  return reg;
}

/// Gets the registers \p generic 's body allocates for itself, in body order.
///
/// A pattern allocates the scratch a template computes in where it matched, so
/// these are in the body as well. Only ones sized entirely by their type are
/// taken: an allocation the body computes a size for could not be moved out of
/// it anyway.
auto getBodyAllocations(linalg::GenericOp generic)
    -> SmallVector<memref::AllocOp> {
  SmallVector<memref::AllocOp> result;
  generic.getBody()->walk([&](memref::AllocOp alloc) {
    if (alloc->getNumOperands() != 0) return;
    result.push_back(alloc);
  });
  return result;
}

/// Hoists the registers of \p generic out of its body.
auto hoistRegisterConstants(linalg::GenericOp generic, AnalysisManager analyses,
                            RewriterBase& rewriter) -> LogicalResult {
  const auto constants = getMappedConstants(generic);

  for (auto constant : constants) {
    // Only one still inside has to move, and only then: moving one that is
    // already out could put it after something else that reads it.
    if (generic->isProperAncestor(constant)) {
      rewriter.moveOpBefore(constant, generic);
    }
    const auto reg = materializeRegister(constant, generic, analyses, rewriter);
    if (!reg) return failure();

    // Whatever read the constant in the body reads the register now. The body
    // is not isolated from above, so it refers to it where it stands.
    rewriter.replaceUsesWithIf(constant.getResult(), reg, [&](OpOperand& use) {
      return generic->isProperAncestor(use.getOwner());
    });
  }

  // The scratch moves out as it stands. What reads it is inside the body and
  // goes on reading it there, which it may: the body is not isolated from
  // above. It cannot be handed in as an operand instead -- a generic takes
  // either tensors or buffers throughout, and the data here is tensors.
  for (auto alloc : getBodyAllocations(generic)) {
    rewriter.moveOpBefore(alloc, generic);
  }

  return success();
}

struct HoistRegistersPass
    : public impl::HoistRegistersPassBase<HoistRegistersPass> {
  using HoistRegistersPassBase::HoistRegistersPassBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    auto result = success();
    getOperation()->walk([&](linalg::GenericOp generic) {
      if (failed(hoistRegisterConstants(generic, getAnalysisManager(),
                                        rewriter))) {
        result = failure();
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });

    if (failed(result)) signalPassFailure();
  }
};

}  // namespace
}  // namespace scheduler
