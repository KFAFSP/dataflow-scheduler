// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances -parallelize-loops-across-instances | FileCheck %s

// Two runs back-to-back. Second run finds no scf.for to transform (the
// new ktdf.parallel is not an scf.for), so output is identical to a single
// run. Verify exactly one ktdf.parallel header appears (the open-paren
// suffix disambiguates from ktdf.parallel_yield in the body).

// CHECK-LABEL: func.func @idempotent
// CHECK:         ktdf.parallel (
// CHECK-NOT:    ktdf.parallel (
// CHECK:         return

func.func @idempotent() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
