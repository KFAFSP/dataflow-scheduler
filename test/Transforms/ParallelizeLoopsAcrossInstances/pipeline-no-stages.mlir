// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Pipeline contains only a ktdf.private (no stages). Effective
// applicable_units is empty. Pipeline is rejected.

// CHECK-LABEL:   func.func @no_stages() {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT:      scf.for %[[IV:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          %[[YIELDED:[a-zA-Z0-9_]+]] = ktdf.private -> (!ktdf.token) {
// CHECK-NEXT:            %[[T:[a-zA-Z0-9_]+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            ktdf.private_yield %[[T]] : !ktdf.token
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @no_stages() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      ktdf.pipeline {
        %t = ktdf.private -> (!ktdf.token) {
          %tk = ktdf.create_token : !ktdf.token
          ktdf.private_yield %tk : !ktdf.token
        }
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}