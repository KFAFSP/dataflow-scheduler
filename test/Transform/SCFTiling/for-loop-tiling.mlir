// RUN: scheduler --tile-scf-for-loops="tile-sizes=32,32" --canonicalize %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (-s0 + s1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>


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
// CHECK-NEXT:         %[[VAL_10:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_6]], %[[VAL_9]]]
// CHECK-NEXT:         scf.for %[[VAL_11:.*]] = %[[VAL_1]] to %[[VAL_10]] step %[[VAL_4]] {
// CHECK-NEXT:           %[[VAL_12:.*]] = arith.addi %[[VAL_7]], %[[VAL_5]] : index
// CHECK-NEXT:           %[[VAL_13:.*]] = arith.minsi %[[VAL_12]], %[[VAL_3]] : index
// CHECK-NEXT:           %[[VAL_14:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_7]], %[[VAL_13]]]
// CHECK-NEXT:           scf.for %[[VAL_15:.*]] = %[[VAL_1]] to %[[VAL_14]] step %[[VAL_4]] {
// CHECK-NEXT:             %[[VAL_16:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_15]]){{\[}}%[[VAL_7]]]
// CHECK-NEXT:             %[[VAL_17:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_11]]){{\[}}%[[VAL_6]]]
// CHECK-NEXT:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_16]] : index
// CHECK-NEXT:             %[[VAL_19:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_15]]){{\[}}%[[VAL_7]]]
// CHECK-NEXT:             %[[VAL_20:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_11]]){{\[}}%[[VAL_6]]]
// CHECK-NEXT:             memref.store %[[VAL_18]], %[[VAL_0]]{{\[}}%[[VAL_20]], %[[VAL_19]]] : memref<10x200xindex>
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
