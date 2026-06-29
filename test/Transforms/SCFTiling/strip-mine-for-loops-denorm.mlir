// RUN: dataflow-scheduler-opt --strip-mine-scf-for-loops="strip-mine-size=32 normalize-loops=false" --canonicalize %s | FileCheck %s

// CHECK-LABEL:   func.func @single_loop(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10xindex>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] {
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[VAL_0]] to %[[MINSI_0]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         %[[ADDI_1:.*]] = arith.addi %[[VAL_1]], %[[VAL_1]] : index
// CHECK-NEXT:         memref.store %[[ADDI_1]], %[[ARG0]]{{\[}}%[[VAL_1]]] : memref<10xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



module {
  func.func @single_loop(%arg0: memref<10xindex>) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    
    scf.for %i = %c0 to %c100 step %c1 {
      // Some computation
      %sum = arith.addi %i, %i : index
      memref.store %sum, %arg0[%i] : memref<10xindex>
    }
    return
  }
}
