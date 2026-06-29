// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Outer trip 5 (not divisible), inner trip 8 (divisible). Pass 1 walks
// outermost-first, skips outer (rule 4 fails), accepts inner. The outer
// scf.for survives; the inner scf.for becomes ktdf.parallel.

// CHECK-LABEL:   func.func @inner_divisible() {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C5:.+]] = arith.constant 5 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      scf.for %[[OUT:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C5]] step %[[C1]] {
// CHECK-NEXT:        ktdf.parallel (%[[IN:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%[[C0]]) to (%[[C8]]) step (%[[C1]]) distribute(num_instances = 2) {
// CHECK-NEXT:          ktdf.pipeline {
// CHECK-NEXT:            ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:              "test.body"(%[[OUT]], %[[IN]]) : (index, index) -> ()
// CHECK-NEXT:            } {applicable_units = ["SFU"]}
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.parallel_yield
// CHECK-NEXT:        }
// CHECK-NEXT:      } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @inner_divisible() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c5 step %c1 {
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