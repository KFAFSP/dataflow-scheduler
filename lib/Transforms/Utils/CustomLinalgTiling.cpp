//===----------------------------------------------------------------------===//
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
// Custom tiling without iter_args.
// Exact copy of upstream tileLinalgOpImpl with 3 modifications:
//   1. Remove insertSlicesBack call
//   2. Return empty scf::ValueVector (no iter_args)
//   3. Return tensorResults directly (not from loop)
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/Utils/CustomLinalgTiling.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"

// NOTE: These are _not_ the parent namespaces, but we want to preserve the
//       spelling of this file to keep it in synch with upstream!
using namespace mlir;
using namespace mlir::linalg;

template <typename LoopTy>
static FailureOr<TiledLinalgOp> tileLinalgOpImpl(
    RewriterBase& b, LinalgOp op, ArrayRef<OpFoldResult> tileSizes,
    const LinalgTilingOptions& options) {
  OpBuilder::InsertionGuard g(b);

  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      })) {
    TiledLinalgOp tiledOp;
    tiledOp.op = cast<LinalgOp>(b.clone(*op.getOperation()));
    tiledOp.tensorResults.assign(tiledOp.op->result_begin(),
                                 tiledOp.op->result_end());
    return tiledOp;
  }

  // 1. Build the tiled loop ranges.
  SmallVector<OpFoldResult> allShapeSizes =
      op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  assert(shapeSizesToLoopsMap && "invalid linalgOp with null ShapesToLoopsMap");

  auto [loopRanges, loopIndexToRangeIndex] = makeTiledLoopRanges(
      b, op.getLoc(), shapeSizesToLoopsMap, allShapeSizes, tileSizes);

  SmallVector<utils::IteratorType, 4> iteratorTypes;
  for (const auto& attr : enumerate(op.getIteratorTypesArray())) {
    if (loopIndexToRangeIndex.count(attr.index()))
      iteratorTypes.push_back(attr.value());
  }
  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
      AffineMap::getMultiDimIdentityMap(tileSizes.size(), b.getContext());
  if (!options.interchangeVector.empty()) {
    // Based on the pruned iterations (due to zero tile size), recompute the
    // interchange vector.
    SmallVector<unsigned, 4> interchangeVector;
    interchangeVector.reserve(options.interchangeVector.size());
    for (auto pos : options.interchangeVector) {
      auto it = loopIndexToRangeIndex.find(pos);
      if (it == loopIndexToRangeIndex.end()) continue;
      interchangeVector.push_back(it->second);
    }
    // Interchange vector is guaranteed to be a permutation,
    // `inversePermutation` must succeed.
    invPermutationMap = inversePermutation(
        AffineMap::getPermutationMap(interchangeVector, b.getContext()));
    assert(invPermutationMap);
    SmallVector<int64_t> permutation(interchangeVector.begin(),
                                     interchangeVector.end());
    applyPermutationToVector(loopRanges, permutation);
    applyPermutationToVector(iteratorTypes, permutation);
  }

  // Handle distribution. Create a vector of the same size of loops that are to
  // be tiled.
  SmallVector<linalg::ProcInfo> procInfo;
  if (options.distribution) {
    procInfo.resize(
        iteratorTypes.size(),
        linalg::ProcInfo{nullptr, nullptr, linalg::DistributionMethod::None});
    // Collect loop ranges of tiled loops, loops that are parallel.
    SmallVector<Range> parallelLoopRanges;
    for (const auto& iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value())) break;
      parallelLoopRanges.push_back(loopRanges[iteratorType.index()]);
    }
    auto returnedProcInfo =
        options.distribution->procInfo(b, op.getLoc(), parallelLoopRanges);
    unsigned procIdIdx = 0;
    // Update the distribution information for the loops.
    for (const auto& iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value())) break;
      procInfo[iteratorType.index()] = returnedProcInfo[procIdIdx++];
    }
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto tiledLoopBodyBuilder =
      [&](OpBuilder& builder, Location loc, ValueRange localIvs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
    ivs.assign(localIvs.begin(), localIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<Value, 4> interchangedIvs;
    if (!options.interchangeVector.empty()) {
      for (AffineExpr result : invPermutationMap.getResults())
        interchangedIvs.push_back(
            ivs[cast<AffineDimExpr>(result).getPosition()]);
    } else {
      interchangedIvs.assign(ivs.begin(), ivs.end());
    }

    // Tile the `operandValuesToUse` that either match the `op` operands
    // themselves or the tile loop arguments forwarding them.
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(op->getNumOperands()) &&
           "expect the number of operands and inputs and outputs to match");
    SmallVector<Value> valuesToTile = operandValuesToUse;
    SmallVector<OpFoldResult> sizeBounds =
        mlir::affine::makeComposedFoldedMultiResultAffineApply(
            b, loc, shapeSizesToLoopsMap, allShapeSizes);
    SmallVector<Value> tiledOperands = makeTiledShapes(
        b, loc, op, valuesToTile, getAsOpFoldResult(interchangedIvs), tileSizes,
        sizeBounds,
        /*omitPartialTileCheck=*/false);

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(op, tiledOperands);
    res = clone(b, op, resultTensorTypes, tiledOperands);

    // ****************************************************************************
    // MODIFICATION: Don't call insertSlicesBack, just store tiled op results
    // ****************************************************************************
    tensorResults.assign(res->result_begin(), res->result_end());

    // ****************************************************************************
    // MODIFICATION: Return empty - no iter_args!
    // ****************************************************************************
    return scf::ValueVector();
  };

  // ****************************************************************************
  // MODIFICATION: Build loop nest WITHOUT iter_args
  // GenerateLoopNest automatically adds op operands as iter_args, so bypass it.
  // ****************************************************************************
  Location loc = op.getLoc();
  SmallVector<Value> lbs, ubs, steps;
  for (Range r : loopRanges) {
    lbs.push_back(getValueOrCreateConstantIndexOp(b, loc, r.offset));
    ubs.push_back(getValueOrCreateConstantIndexOp(b, loc, r.size));
    steps.push_back(getValueOrCreateConstantIndexOp(b, loc, r.stride));
  }

  SmallVector<Value> emptyIterArgs;
  scf::buildLoopNest(b, loc, lbs, ubs, steps, emptyIterArgs,
                     [&](OpBuilder& nestedBuilder, Location nestedLoc,
                         ValueRange localIvs, ValueRange iterArgs) {
                       // Call body builder with the original operands (not as
                       // iter_args)
                       return tiledLoopBodyBuilder(nestedBuilder, nestedLoc,
                                                   localIvs, op->getOperands());
                     });

  // 3. Transform IndexOp results w.r.t. the tiling.
  transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

  // 4. Gather the newly created loops and return them with the new op.
  SmallVector<Operation*, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    if (isa<BlockArgument>(iv)) {
      loops.push_back(cast<BlockArgument>(iv).getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      // TODO: Instead of doing this, try to recover the ops used instead of the
      // loop.
      loops.push_back(nullptr);
    }
  }

  // ****************************************************************************
  // MODIFICATION: Return tensorResults directly (not results of outermost loop)
  // ****************************************************************************
  return TiledLinalgOp{res, loops, tensorResults};
}

FailureOr<TiledLinalgOp> scheduler::customTileLinalgOp(
    RewriterBase& rewriter, LinalgOp op, const LinalgTilingOptions& options) {
  SmallVector<Value> tileSizeValues =
      options.tileSizeComputationFunction(rewriter, op);
  SmallVector<OpFoldResult> tileSizes = getAsOpFoldResult(tileSizeValues);
  return tileLinalgOpImpl<scf::ForOp>(rewriter, op, tileSizes, options);
}

// Made with Bob
