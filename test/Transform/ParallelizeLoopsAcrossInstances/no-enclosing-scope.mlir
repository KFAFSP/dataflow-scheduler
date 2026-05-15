// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Pipeline at function top-level, no scf.for parents. No candidate.

// CHECK-LABEL: func.func @no_scope
// CHECK:         ktdf.pipeline {
// CHECK-NOT:    ktdf.parallel
// CHECK:         }

func.func @no_scope() {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      "test.body"() : () -> ()
    } {applicable_units = #ktdf.applicable_units<[SFP]>}
  }
  return
}
