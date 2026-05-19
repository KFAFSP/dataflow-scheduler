// RUN: not scheduler %s 2>&1 | FileCheck %s

// Negative: modulo size < 2 must be rejected by the verifier.

// CHECK: 'ktdf.pipeline' op modulo size must be >= 2, got 1

func.func @pipeline_modulo_too_small() {
  ktdf.pipeline modulo(size: 1) {
    ktdf.stage depends_in(none) depends_out(none) {
      %c0 = arith.constant 0 : index
    }
  }
  return
}
