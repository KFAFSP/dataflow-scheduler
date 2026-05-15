// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances -debug-only=parallelize-loops-across-instances 2>&1 | FileCheck %s --check-prefix=DEBUG

// Verify debug log emits the imbalanced-fallback line with the exact
// trip count, num_instances, and the per-instance chunk sizes.

// DEBUG:      [parallelize-loops-across-instances] num_instances=2
// DEBUG-NEXT: [parallelize-loops-across-instances] imbalanced (fallback) at loc({{.*}}): trip=5, num_instances=2, chunks=3,2

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
