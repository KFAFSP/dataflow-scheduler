// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Loop in scope is reduction, not parallel. No candidate.

// CHECK-LABEL: func.func @reduction_loop
// CHECK:         scf.for
// CHECK-NOT:    ktdf.parallel
// CHECK:         {loop_type = #ktdf.loop_type<reduction_loop>}

func.func @reduction_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<reduction_loop>}
  return
}
