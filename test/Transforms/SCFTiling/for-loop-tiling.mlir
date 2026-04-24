// RUN: scheduler --tile-scf-for-loops="tile-sizes=32,32" %s | FileCheck %s




// CHECK-LABEL:   func.func @nested_for_loops() {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 200 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.muli %[[VAL_3]], %[[VAL_4]] : index
// CHECK-NEXT:     scf.for %[[VAL_7:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_6]] {
// CHECK-NEXT:       %[[VAL_8:.*]] = arith.muli %[[VAL_3]], %[[VAL_5]] : index
// CHECK-NEXT:       scf.for %[[VAL_9:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_8]] {
// CHECK-NEXT:         %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_6]] : index
// CHECK-NEXT:         %[[VAL_11:.*]] = arith.minsi %[[VAL_1]], %[[VAL_10]] : index
// CHECK-NEXT:         scf.for %[[VAL_12:.*]] = %[[VAL_7]] to %[[VAL_11]] step %[[VAL_3]] {
// CHECK-NEXT:           %[[VAL_13:.*]] = arith.addi %[[VAL_9]], %[[VAL_8]] : index
// CHECK-NEXT:           %[[VAL_14:.*]] = arith.minsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK-NEXT:           scf.for %[[VAL_15:.*]] = %[[VAL_9]] to %[[VAL_14]] step %[[VAL_3]] {
// CHECK-NEXT:             %[[VAL_16:.*]] = arith.addi %[[VAL_12]], %[[VAL_15]] : index
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }









module {
func.func @nested_for_loops() {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %c100 step %c1 {
    scf.for %j = %c0 to %c200 step %c1 {
      // Some computation
      %sum = arith.addi %i, %j : index
    }
  }
  return
}
}
