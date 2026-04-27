// RUN: scheduler --strip-mine-scf-for-loops="strip-mine-size=32 normalize-inner-loop=false" --canonicalize %s | FileCheck %s


// CHECK-LABEL:   func.func @single_loop(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<10xindex>) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:       %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_4]] : index
// CHECK-NEXT:       %[[VAL_7:.*]] = arith.minsi %[[VAL_6]], %[[VAL_2]] : index
// CHECK-NEXT:       scf.for %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_3]] {
// CHECK-NEXT:         %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_8]] : index
// CHECK-NEXT:         memref.store %[[VAL_9]], %[[VAL_0]]{{\[}}%[[VAL_8]]] : memref<10xindex>
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
