// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Loop bounds depend on a function argument. Trip count not statically
// known. No candidate.

// CHECK-LABEL:   func.func @dynamic_trip(
// CHECK-SAME:        %[[UB:[a-zA-Z0-9_]+]]: index) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      scf.for %[[IV:[a-zA-Z0-9_]+]] = %[[C0]] to %[[UB]] step %[[C1]] {
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:            "test.body"(%[[IV]]) : (index) -> ()
// CHECK-NEXT:          } {applicable_units = #ktdf.applicable_units<[SFP]>}
// CHECK-NEXT:        }
// CHECK-NEXT:      } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      return
// CHECK-NEXT:    }

func.func @dynamic_trip(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %arg0 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
