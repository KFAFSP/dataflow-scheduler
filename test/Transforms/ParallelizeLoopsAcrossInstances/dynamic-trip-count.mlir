// RUN: dataflow-scheduler-opt -allow-unregistered-dialect  -parallelize-loops-across-instances %s | FileCheck %s

// CHECK-LABEL:   func.func @dynamic_trip(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     ktdf.parallel (%[[VAL_0:.*]], %[[VAL_1:.*]]) = (%[[CONSTANT_0]]) to (%[[ARG0]]) step (%[[CONSTANT_1]]) distribute(num_instances = 2) {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:           "test.body"(%[[VAL_0]]) : (index) -> ()
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.parallel_yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Loop bounds depend on a function argument. Trip count not statically
// known but still a candidate.

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @dynamic_trip(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %arg0 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i) : (index) -> ()
        } {applicable_units = ["SFU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}