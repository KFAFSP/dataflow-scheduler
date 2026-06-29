// RUN: dataflow-scheduler-opt --tile-scf-for-loops="tile-sizes=32,32 normalize-loops=false fallback-scf-tiling=true" --canonicalize %s | FileCheck %s


// CHECK-LABEL:   func.func @nested_for_loops(
// CHECK-SAME:                                %[[VAL_0:.*]]: memref<10x200xindex>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 200 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_6:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_5]] {
// CHECK-NEXT:       scf.for %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-NEXT:         %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_5]] : index
// CHECK-NEXT:         %[[VAL_9:.*]] = arith.minsi %[[VAL_8]], %[[VAL_2]] : index
// CHECK-NEXT:         scf.for %[[VAL_10:.*]] = %[[VAL_6]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK-NEXT:           %[[VAL_11:.*]] = arith.addi %[[VAL_7]], %[[VAL_5]] : index
// CHECK-NEXT:           %[[VAL_12:.*]] = arith.minsi %[[VAL_11]], %[[VAL_3]] : index
// CHECK-NEXT:           scf.for %[[VAL_13:.*]] = %[[VAL_7]] to %[[VAL_12]] step %[[VAL_4]] {
// CHECK-NEXT:             %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_13]] : index
// CHECK-NEXT:             memref.store %[[VAL_14]], %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_13]]] : memref<10x200xindex>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
func.func @nested_for_loops(%A : memref<10x200xindex>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %c100 step %c1 {
    scf.for %j = %c0 to %c200 step %c1 {
      // Some computation
      %sum = arith.addi %i, %j : index
      memref.store %sum, %A[%i, %j] : memref<10x200xindex>
    }
  }
  return
}
}
