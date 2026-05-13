// RUN: scheduler %s -broadcast-promotion | FileCheck %s

// CHECK-LABEL: func.func @trivial
func.func @trivial() {
  return
}
