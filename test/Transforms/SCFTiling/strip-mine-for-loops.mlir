// RUN: scheduler --strip-mine-scf-for-loops="strip-mine-size=32" %s | FileCheck %s




// CHECK-LABEL:   func.func @single_loop() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.muli %[[VAL_2]], %[[VAL_3]] : index
// CHECK-NEXT:     scf.for %[[VAL_5:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_4]] {
// CHECK-NEXT:       %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_4]] : index
// CHECK-NEXT:       %[[VAL_7:.*]] = arith.minsi %[[VAL_1]], %[[VAL_6]] : index
// CHECK-NEXT:       scf.for %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_2]] {
// CHECK-NEXT:         %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_8]] : index
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @independent_loops() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 50 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.muli %[[VAL_2]], %[[VAL_3]] : index
// CHECK-NEXT:     scf.for %[[VAL_5:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_4]] {
// CHECK-NEXT:       %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_4]] : index
// CHECK-NEXT:       %[[VAL_7:.*]] = arith.minsi %[[VAL_1]], %[[VAL_6]] : index
// CHECK-NEXT:       scf.for %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_2]] {
// CHECK-NEXT:         %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_8]] : index
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[VAL_10:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_11:.*]] = arith.muli %[[VAL_2]], %[[VAL_10]] : index
// CHECK-NEXT:     scf.for %[[VAL_12:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_11]] {
// CHECK-NEXT:       %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_11]] : index
// CHECK-NEXT:       %[[VAL_14:.*]] = arith.minsi %[[VAL_1]], %[[VAL_13]] : index
// CHECK-NEXT:       scf.for %[[VAL_15:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_2]] {
// CHECK-NEXT:         %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[VAL_15]] : index
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @nested_loops() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 200 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.muli %[[VAL_3]], %[[VAL_4]] : index
// CHECK-NEXT:     scf.for %[[VAL_6:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_5]] {
// CHECK-NEXT:       %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_5]] : index
// CHECK-NEXT:       %[[VAL_8:.*]] = arith.minsi %[[VAL_1]], %[[VAL_7]] : index
// CHECK-NEXT:       scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_3]] {
// CHECK-NEXT:         %[[VAL_10:.*]] = arith.constant 32 : index
// CHECK-NEXT:         %[[VAL_11:.*]] = arith.muli %[[VAL_3]], %[[VAL_10]] : index
// CHECK-NEXT:         scf.for %[[VAL_12:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_11]] {
// CHECK-NEXT:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_11]] : index
// CHECK-NEXT:           %[[VAL_14:.*]] = arith.minsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK-NEXT:           scf.for %[[VAL_15:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_3]] {
// CHECK-NEXT:             %[[VAL_16:.*]] = arith.addi %[[VAL_9]], %[[VAL_15]] : index
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



module {
func.func @single_loop() {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %c100 step %c1 {
    // Some computation
    %sum = arith.addi %i, %i : index
  }
  return
}

func.func @independent_loops() {
  %c0 = arith.constant 0 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  
  // First loop should be strip-mined
  scf.for %i = %c0 to %c50 step %c1 {
    %val = arith.addi %i, %i : index
  }
  
  // Second independent loop should also be strip-mined
  scf.for %j = %c0 to %c50 step %c1 {
    %val = arith.muli %j, %j : index
  }
  
  return
}

func.func @nested_loops() {
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
    }
  }
  return
}
}