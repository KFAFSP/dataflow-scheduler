// RUN: scheduler %s | scheduler | FileCheck %s

// Round-trip: pipeline without modulo (existing behavior).
// CHECK-LABEL:   func.func @pipeline_no_modulo
// CHECK:           ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:          arith.constant 0 : index
// CHECK-NEXT:        }
// CHECK-NEXT:      }

func.func @pipeline_no_modulo() {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %c0 = arith.constant 0 : index
    }
  }
  return
}

// Round-trip: pipeline with modulo(size: 2).
// CHECK-LABEL:   func.func @pipeline_modulo_2
// CHECK:           ktdf.pipeline modulo(size : 2) {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:          arith.constant 0 : index
// CHECK-NEXT:        }
// CHECK-NEXT:      }

func.func @pipeline_modulo_2() {
  ktdf.pipeline modulo(size: 2) {
    ktdf.stage depends_in(none) depends_out(none) {
      %c0 = arith.constant 0 : index
    }
  }
  return
}

// Round-trip: pipeline with modulo(size: 4) — N>2 still parses.
// CHECK-LABEL:   func.func @pipeline_modulo_4
// CHECK:           ktdf.pipeline modulo(size : 4) {

func.func @pipeline_modulo_4() {
  ktdf.pipeline modulo(size: 4) {
    ktdf.stage depends_in(none) depends_out(none) {
      %c0 = arith.constant 0 : index
    }
  }
  return
}
