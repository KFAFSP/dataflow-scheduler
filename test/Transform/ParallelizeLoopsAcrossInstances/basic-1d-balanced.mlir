// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// One pipeline, corelet-local applicable_units, scope with one parallel
// loop, trip count 4 (balanced split for num_instances=2).

// CHECK-LABEL: func.func @basic_1d_balanced
// CHECK:         ktdf.parallel (%[[IV:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) distribute(num_instances = 2) {
// CHECK:           "test.body"(%[[IV]]) : (index) -> ()
// CHECK:           ktdf.parallel_yield
// CHECK:         }

func.func @basic_1d_balanced() {
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
