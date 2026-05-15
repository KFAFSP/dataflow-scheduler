// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Pipeline contains only a ktdf.private (no stages). Effective
// applicable_units is empty. Pipeline is rejected.

// CHECK-LABEL: func.func @no_stages
// CHECK:         scf.for
// CHECK-NOT:    ktdf.parallel

func.func @no_stages() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      %t = ktdf.private -> (!ktdf.token) {
        %tk = ktdf.create_token : !ktdf.token
        ktdf.private_yield %tk : !ktdf.token
      }
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
