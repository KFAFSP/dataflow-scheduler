// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// applicable_units includes L3LU (core-shared). Pipeline is rejected; loop
// stays as scf.for.

// CHECK-LABEL: func.func @non_corelet
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           ktdf.pipeline {
// CHECK-NOT:    ktdf.parallel
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

func.func @non_corelet() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[L3LU]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
