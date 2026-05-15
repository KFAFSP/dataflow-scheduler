// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Two parallel loops in scope, both divisible by 2. Outermost (trip 4) wins;
// inner (trip 8) remains an scf.for inside the parallel body.

// CHECK-LABEL: func.func @outermost_wins
// CHECK:         ktdf.parallel (%[[OUT:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) distribute(num_instances = 2) {
// CHECK:           scf.for %[[IN:[a-zA-Z0-9_]+]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             "test.body"(%[[OUT]], %[[IN]]) : (index, index) -> ()
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:           ktdf.parallel_yield
// CHECK:         }

func.func @outermost_wins() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i, %j) : (index, index) -> ()
        } {applicable_units = #ktdf.applicable_units<[SFP]>}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
