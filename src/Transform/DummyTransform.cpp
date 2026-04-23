#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"

namespace ktdf {
namespace {

#define GEN_PASS_DEF_DUMMYTRANSFORMPASS
#include "src/Transform/Passes.h.inc"

struct DummyTransformPass
    : public impl::DummyTransformPassBase<DummyTransformPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());

    mlir::Value token1 =
        mlir::ktdf::CreateTokenOp::create(builder, module.getLoc()).getResult();

    auto pipeline = mlir::ktdf::PipelineOp::create(builder, module.getLoc());

    builder.setInsertionPointToStart(pipeline.getBodyBlock());
    mlir::ktdf::StageOp::create(builder, module.getLoc(),
                                mlir::ValueRange{token1}, mlir::ValueRange{});
    mlir::ktdf::StageOp::create(builder, module.getLoc(), mlir::ValueRange{},
                                mlir::ValueRange{token1});
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createDummyTransformPass() {
  return std::make_unique<DummyTransformPass>();
}

}  // namespace ktdf
