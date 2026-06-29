// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Two parallel loops in scope, both divisible by 2. Outermost (trip 4) wins;
// inner (trip 8) remains an scf.for inside the parallel body.

// CHECK-LABEL:   func.func @outermost_wins() {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      ktdf.parallel (%[[OUT:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%[[C0]]) to (%[[C4]]) step (%[[C1]]) distribute(num_instances = 2) {
// CHECK-NEXT:        scf.for %[[IN:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-NEXT:          ktdf.pipeline {
// CHECK-NEXT:            ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:              "test.body"(%[[OUT]], %[[IN]]) : (index, index) -> ()
// CHECK-NEXT:            } {applicable_units = ["SFU"]}
// CHECK-NEXT:          }
// CHECK-NEXT:        } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }


module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
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
          } {applicable_units = ["SFU"]}
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}