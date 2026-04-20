#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Transform/Passes.h"

namespace ktdf {
namespace {

#define GEN_PASS_DEF_DUMMYTRANSFORMPASS
#include "src/Transform/Passes.h.inc"

struct DummyTransformPass
    : public impl::DummyTransformPassBase<DummyTransformPass> {
  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<mlir::Pass> createDummyTransformPass() {
  return std::make_unique<DummyTransformPass>();
}

}  // namespace ktdf
