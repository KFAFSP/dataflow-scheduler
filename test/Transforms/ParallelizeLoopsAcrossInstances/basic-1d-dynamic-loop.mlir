// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// One pipeline, corelet-local applicable_units, scope with one parallel
// loop with dynamic upper bound (balance unknown).

// CHECK-LABEL:   func.func @basic_1d_dynamic_loop() {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT:      %[[V0:.+]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:      %[[V1:.+]] = arith.ceildivui %[[C4]], %[[V0]] : index
// CHECK-NEXT:      ktdf.parallel (%[[IV:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%[[C0]]) to (%[[V1]]) step (%[[C1]]) distribute(num_instances = 2) {
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:            "test.body"(%[[IV]]) : (index) -> ()
// CHECK-NEXT:          } {applicable_units = ["SFU"]}
// CHECK-NEXT:        }
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @basic_1d_dynamic_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %4 = arith.ceildivui %c4, %3 : index
    scf.for %i = %c0 to %4 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i) : (index) -> ()
        } {applicable_units = ["SFU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
