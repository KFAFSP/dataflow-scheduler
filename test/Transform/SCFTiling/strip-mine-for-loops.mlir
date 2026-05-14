// RUN: scheduler --strip-mine-scf-for-loops="strip-mine-size=32" --canonicalize %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (-s0 + s1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>




// CHECK-LABEL:   func.func @single_loop(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<10xindex>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:       %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_4]] : index
// CHECK-NEXT:       %[[VAL_7:.*]] = arith.minsi %[[VAL_6]], %[[VAL_2]] : index
// CHECK-NEXT:       %[[VAL_8:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_5]], %[[VAL_7]]]
// CHECK-NEXT:       scf.for %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_3]] {
// CHECK-NEXT:         %[[VAL_10:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_9]]){{\[}}%[[VAL_5]]]
// CHECK-NEXT:         %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_10]] : index
// CHECK-NEXT:         memref.store %[[VAL_11]], %[[VAL_0]]{{\[}}%[[VAL_10]]] : memref<10xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @independent_loops(
// CHECK-SAME:                                 %[[VAL_0:.*]]: memref<10xindex>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 50 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:       %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_4]] : index
// CHECK-NEXT:       %[[VAL_7:.*]] = arith.minsi %[[VAL_6]], %[[VAL_2]] : index
// CHECK-NEXT:       %[[VAL_8:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_5]], %[[VAL_7]]]
// CHECK-NEXT:       scf.for %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_3]] {
// CHECK-NEXT:         %[[VAL_10:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_9]]){{\[}}%[[VAL_5]]]
// CHECK-NEXT:         %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_10]] : index
// CHECK-NEXT:         memref.store %[[VAL_11]], %[[VAL_0]]{{\[}}%[[VAL_10]]] : memref<10xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.for %[[VAL_12:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:       %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_4]] : index
// CHECK-NEXT:       %[[VAL_14:.*]] = arith.minsi %[[VAL_13]], %[[VAL_2]] : index
// CHECK-NEXT:       %[[VAL_15:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_12]], %[[VAL_14]]]
// CHECK-NEXT:       scf.for %[[VAL_16:.*]] = %[[VAL_1]] to %[[VAL_15]] step %[[VAL_3]] {
// CHECK-NEXT:         %[[VAL_17:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_16]]){{\[}}%[[VAL_12]]]
// CHECK-NEXT:         %[[VAL_18:.*]] = arith.muli %[[VAL_17]], %[[VAL_17]] : index
// CHECK-NEXT:         memref.store %[[VAL_18]], %[[VAL_0]]{{\[}}%[[VAL_17]]] : memref<10xindex>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @nested_loops(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<10x200xindex>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 200 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_6:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_5]] {
// CHECK-NEXT:       %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_5]] : index
// CHECK-NEXT:       %[[VAL_8:.*]] = arith.minsi %[[VAL_7]], %[[VAL_2]] : index
// CHECK-NEXT:       %[[VAL_9:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_6]], %[[VAL_8]]]
// CHECK-NEXT:       scf.for %[[VAL_10:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK-NEXT:         %[[VAL_11:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_10]]){{\[}}%[[VAL_6]]]
// CHECK-NEXT:         scf.for %[[VAL_12:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-NEXT:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_5]] : index
// CHECK-NEXT:           %[[VAL_14:.*]] = arith.minsi %[[VAL_13]], %[[VAL_3]] : index
// CHECK-NEXT:           %[[VAL_15:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_12]], %[[VAL_14]]]
// CHECK-NEXT:           scf.for %[[VAL_16:.*]] = %[[VAL_1]] to %[[VAL_15]] step %[[VAL_4]] {
// CHECK-NEXT:             %[[VAL_17:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_16]]){{\[}}%[[VAL_12]]]
// CHECK-NEXT:             %[[VAL_18:.*]] = arith.addi %[[VAL_11]], %[[VAL_17]] : index
// CHECK-NEXT:             memref.store %[[VAL_18]], %[[VAL_0]]{{\[}}%[[VAL_11]], %[[VAL_17]]] : memref<10x200xindex>
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
