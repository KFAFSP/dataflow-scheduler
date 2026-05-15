// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Pipeline at function top-level, no scf.for parents. No candidate.

// CHECK-LABEL:   func.func @no_scope() {
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:          "test.body"() : () -> ()
// CHECK-NEXT:        } {applicable_units = #ktdf.applicable_units<[SFP]>}
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

func.func @no_scope() {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      "test.body"() : () -> ()
    } {applicable_units = #ktdf.applicable_units<[SFP]>}
  }
  return
}
