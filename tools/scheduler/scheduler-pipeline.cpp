#include "Transform/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace ktdf {

void buildKTDPToDFIRPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createDummyTransformPass());
}

}  // namespace ktdf
