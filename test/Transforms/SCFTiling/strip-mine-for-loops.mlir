// RUN: dataflow-scheduler-opt --strip-mine-scf-for-loops="strip-mine-size=32" --canonicalize %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (-s0 + s1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL:   func.func @single_loop(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10xindex>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] {
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_0]], %[[MINSI_0]]]
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[APPLY_0]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:         %[[APPLY_2:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:         %[[ADDI_1:.*]] = arith.addi %[[APPLY_2]], %[[APPLY_1]] : index
// CHECK-NEXT:         %[[APPLY_3:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:         memref.store %[[ADDI_1]], %[[ARG0]]{{\[}}%[[APPLY_3]]] : memref<10xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @independent_loops(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10xindex>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 50 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] {
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_0]], %[[MINSI_0]]]
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[APPLY_0]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:         %[[APPLY_2:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:         %[[ADDI_1:.*]] = arith.addi %[[APPLY_2]], %[[APPLY_1]] : index
// CHECK-NEXT:         %[[APPLY_3:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:         memref.store %[[ADDI_1]], %[[ARG0]]{{\[}}%[[APPLY_3]]] : memref<10xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.for %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] {
// CHECK-NEXT:       %[[ADDI_2:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_3]] : index
// CHECK-NEXT:       %[[MINSI_1:.*]] = arith.minsi %[[ADDI_2]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[APPLY_4:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_2]], %[[MINSI_1]]]
// CHECK-NEXT:       scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[APPLY_4]] step %[[CONSTANT_2]] {
// CHECK-NEXT:         %[[APPLY_5:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_3]]){{\[}}%[[VAL_2]]]
// CHECK-NEXT:         %[[APPLY_6:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_3]]){{\[}}%[[VAL_2]]]
// CHECK-NEXT:         %[[MULI_0:.*]] = arith.muli %[[APPLY_6]], %[[APPLY_5]] : index
// CHECK-NEXT:         %[[APPLY_7:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_3]]){{\[}}%[[VAL_2]]]
// CHECK-NEXT:         memref.store %[[MULI_0]], %[[ARG0]]{{\[}}%[[APPLY_7]]] : memref<10xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @nested_loops(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10x200xindex>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 200 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_4]] {
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_0]], %[[MINSI_0]]]
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[APPLY_0]] step %[[CONSTANT_3]] {
// CHECK-NEXT:         scf.for %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] {
// CHECK-NEXT:           %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_4]] : index
// CHECK-NEXT:           %[[MINSI_1:.*]] = arith.minsi %[[ADDI_1]], %[[CONSTANT_2]] : index
// CHECK-NEXT:           %[[APPLY_1:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_2]], %[[MINSI_1]]]
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[APPLY_1]] step %[[CONSTANT_3]] {
// CHECK-NEXT:             %[[APPLY_2:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:             %[[APPLY_3:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_3]]){{\[}}%[[VAL_2]]]
// CHECK-NEXT:             %[[ADDI_2:.*]] = arith.addi %[[APPLY_2]], %[[APPLY_3]] : index
// CHECK-NEXT:             %[[APPLY_4:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_1]]){{\[}}%[[VAL_0]]]
// CHECK-NEXT:             %[[APPLY_5:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_3]]){{\[}}%[[VAL_2]]]
// CHECK-NEXT:             memref.store %[[ADDI_2]], %[[ARG0]]{{\[}}%[[APPLY_4]], %[[APPLY_5]]] : memref<10x200xindex>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
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

func.func @independent_loops(%arg0: memref<10xindex>) {
  %c0 = arith.constant 0 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  
  // First loop should be strip-mined
  scf.for %i = %c0 to %c50 step %c1 {
    %val = arith.addi %i, %i : index
    memref.store %val, %arg0[%i] : memref<10xindex>
  }
  
  // Second independent loop should also be strip-mined
  scf.for %j = %c0 to %c50 step %c1 {
    %val = arith.muli %j, %j : index
    memref.store %val, %arg0[%j] : memref<10xindex>
  }
  
  return
}

func.func @nested_loops(%arg0: memref<10x200xindex>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c1 = arith.constant 1 : index
  
  // Both outer and inner loops should be strip-mined independently
  // Outer loop (i) goes from 0 to 100, gets strip-mined with step 32
  // Inner loop (j) goes from 0 to 200, gets strip-mined with step 32
  scf.for %i = %c0 to %c100 step %c1 {
    scf.for %j = %c0 to %c200 step %c1 {
      %sum = arith.addi %i, %j : index
      memref.store %sum, %arg0[%i, %j] : memref<10x200xindex>
    }
  }
  return
}
}
