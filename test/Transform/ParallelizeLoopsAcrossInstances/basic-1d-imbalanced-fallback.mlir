// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Trip count 5 (not divisible by 2). Pass 1 finds nothing; Pass 2 accepts.

// CHECK-LABEL:   func.func @imbalanced_fallback() {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C5:.+]] = arith.constant 5 : index
// CHECK-NEXT:      ktdf.parallel (%[[IV:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%[[C0]]) to (%[[C5]]) step (%[[C1]]) distribute(num_instances = 2) {
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:            "test.body"(%[[IV]]) : (index) -> ()
// CHECK-NEXT:          } {applicable_units = #ktdf.applicable_units<[SFP]>}
// CHECK-NEXT:        }
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

func.func @imbalanced_fallback() {
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
