// RUN: dataflow-scheduler-opt --tile-scf-for-loops="normalize-loops=false" %s | FileCheck %s


// CHECK-LABEL:   func.func @test_two_loops_two_tiles(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[RESERVE_SIZE_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[RESERVE_SIZE_1:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[CONSTANT_0]], %[[RESERVE_SIZE_0]] : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG0]] step %[[MULI_0]] {
// CHECK-NEXT:       %[[MULI_1:.*]] = arith.muli %[[CONSTANT_0]], %[[RESERVE_SIZE_1]] : index
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[ARG1]] step %[[MULI_1]] {
// CHECK-NEXT:         %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[MULI_0]] : index
// CHECK-NEXT:         %[[MINSI_0:.*]] = arith.minsi %[[ARG0]], %[[ADDI_0]] : index
// CHECK-NEXT:         scf.for %[[VAL_2:.*]] = %[[VAL_0]] to %[[MINSI_0]] step %[[CONSTANT_0]] {
// CHECK-NEXT:           %[[ADDI_1:.*]] = arith.addi %[[VAL_1]], %[[MULI_1]] : index
// CHECK-NEXT:           %[[MINSI_1:.*]] = arith.minsi %[[ARG1]], %[[ADDI_1]] : index
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[VAL_1]] to %[[MINSI_1]] step %[[CONSTANT_0]] {
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_single_loop_single_tile(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG0]] step %[[CONSTANT_0]] {
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



func.func @test_two_loops_two_tiles(%N: index, %M: index) {
  %c1 = arith.constant 1 : index
  
  // Create two tile size placeholders
  
  scf.for %i = %c1 to %N step %c1 {
    scf.for %j = %c1 to %M step %c1 {
      // Loop body
    } {loop_type = #ktdf.loop_type<parallel_loop>}
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  
  return
}

func.func @test_single_loop_single_tile(%N: index) {
  %c1 = arith.constant 1 : index
  // For now we don't consider single loops as candidates for tiling (we defer to strip-mining for such loops). This choice is arbitrary at this point and should be revisited.
  scf.for %i = %c1 to %N step %c1 {
    // Loop body
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  
  return
}
