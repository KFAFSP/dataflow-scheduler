// RUN: dataflow-scheduler-opt --tile-scf-for-loops="tile-sizes=32,32 fallback-scf-tiling=true" --canonicalize %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0] -> (d0 * -32 + s0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 32)>
// CHECK-LABEL:   func.func @nested_for_loops(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10x200xindex>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 7 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 200 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 4 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_6]] step %[[CONSTANT_4]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_0]] step %[[CONSTANT_4]] {
// CHECK-NEXT:         %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_0]])
// CHECK-NEXT:         %[[ADDI_0:.*]] = arith.addi %[[APPLY_0]], %[[CONSTANT_5]] : index
// CHECK-NEXT:         %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:         %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[MINSI_0]]]
// CHECK-NEXT:         scf.for %[[VAL_2:.*]] = %[[CONSTANT_1]] to %[[APPLY_1]] step %[[CONSTANT_4]] {
// CHECK-NEXT:           %[[APPLY_2:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_1]])
// CHECK-NEXT:           %[[ADDI_1:.*]] = arith.addi %[[APPLY_2]], %[[CONSTANT_5]] : index
// CHECK-NEXT:           %[[MINSI_1:.*]] = arith.minsi %[[ADDI_1]], %[[CONSTANT_3]] : index
// CHECK-NEXT:           %[[APPLY_3:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[MINSI_1]]]
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_1]] to %[[APPLY_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:             %[[APPLY_4:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_3]], %[[VAL_1]])
// CHECK-NEXT:             %[[APPLY_5:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_2]], %[[VAL_0]])
// CHECK-NEXT:             %[[ADDI_2:.*]] = arith.addi %[[APPLY_5]], %[[APPLY_4]] : index
// CHECK-NEXT:             %[[APPLY_6:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_3]], %[[VAL_1]])
// CHECK-NEXT:             %[[APPLY_7:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_2]], %[[VAL_0]])
// CHECK-NEXT:             memref.store %[[ADDI_2]], %[[ARG0]]{{\[}}%[[APPLY_7]], %[[APPLY_6]]] : memref<10x200xindex>
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
