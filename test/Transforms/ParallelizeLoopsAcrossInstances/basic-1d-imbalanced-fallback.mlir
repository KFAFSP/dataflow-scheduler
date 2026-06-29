// RUN: dataflow-scheduler-opt -allow-unregistered-dialect  -parallelize-loops-across-instances %s | FileCheck %s

// Trip count 5 (not divisible by 2). Pass 1 finds nothing; Pass 2 accepts.

// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
// CHECK-LABEL:   func.func @imbalanced_fallback() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 5 : index
// CHECK-NEXT:     ktdf.parallel (%[[VAL_0:.*]], %[[VAL_1:.*]]) = (%[[CONSTANT_0]]) to (%[[CONSTANT_2]]) step (%[[CONSTANT_1]]) distribute(num_instances = 2) {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:           "test.body"(%[[VAL_0]]) : (index) -> ()
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.parallel_yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @imbalanced_fallback() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.for %i = %c0 to %c5 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i) : (index) -> ()
        } {applicable_units = ["SFU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}