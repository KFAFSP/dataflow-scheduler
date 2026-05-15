// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances -debug-only=parallelize-loops-across-instances 2>&1 | FileCheck %s --check-prefix=DEBUG

// Verify debug log emits the balanced-candidate line with the exact
// trip count and num_instances.

// DEBUG:      [parallelize-loops-across-instances] num_instances=2
// DEBUG-NEXT: [parallelize-loops-across-instances] balanced candidate at loc({{.*}}): trip=4, num_instances=2

func.func @debug_balanced() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
