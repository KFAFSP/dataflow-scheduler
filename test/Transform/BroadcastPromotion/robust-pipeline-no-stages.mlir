// RUN: scheduler --broadcast-promotion %s | FileCheck %s

// Pipeline with only a ktdf.private and no stages. Should complete
// without crashing.

// CHECK-LABEL: func.func @no_stages
// CHECK: ktdf.pipeline {
// CHECK:   ktdf.private

module {
  func.func @no_stages() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %c1 step %c1 {
      ktdf.pipeline {
        %a, %t = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
          %buf = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
          %tok = ktdf.create_token : !ktdf.token
          ktdf.private_yield %buf, %tok : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
        }
      }
    }
    return
  }
}
