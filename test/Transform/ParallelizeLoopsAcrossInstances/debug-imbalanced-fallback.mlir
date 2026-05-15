// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances -debug-only=parallelize-loops-across-instances 2>&1 | FileCheck %s --check-prefix=DEBUG

// DEBUG: imbalanced (fallback)
// DEBUG-SAME: trip=5
// DEBUG-SAME: num_instances=2
// DEBUG-SAME: chunks=3,2

func.func @debug_imbalanced() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  scf.for %i = %c0 to %c5 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
