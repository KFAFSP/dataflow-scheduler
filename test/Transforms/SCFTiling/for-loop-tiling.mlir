// RUN: scheduler-opt %s --tile-scf-for-loops="tile-sizes=32,32" | FileCheck %s

// CHECK-LABEL: func @nested_for_loops
func.func @nested_for_loops() {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: scf.for %{{.*}} = %c0 to %c100 step %c32
  // CHECK:   scf.for %{{.*}} = %c0 to %c100 step %c32_0
  // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c1
  // CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c1_1
  scf.for %i = %c0 to %c100 step %c1 {
    scf.for %j = %c0 to %c100 step %c1 {
      // Some computation
      %sum = arith.addi %i, %j : index
    }
  }
  return
}