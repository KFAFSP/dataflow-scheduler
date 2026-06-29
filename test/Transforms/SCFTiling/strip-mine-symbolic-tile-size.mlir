// RUN: dataflow-scheduler-opt --strip-mine-scf-for-loops %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>
// CHECK-LABEL:   func.func @test_strip_mine_with_symbolic_tile(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[RESERVE_SIZE_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[RESERVE_SIZE_0]] : index
// CHECK-NEXT:     %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG0]], %[[MULI_0]]]
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[APPLY_0]] step %[[CONSTANT_3]] {
// CHECK-NEXT:       %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[APPLY_1]], %[[MULI_0]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ARG0]], %[[ADDI_0]] : index
// CHECK-NEXT:       %[[APPLY_2:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       %[[APPLY_3:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_0]]){{\[}}%[[MINSI_0]], %[[MULI_0]]]
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[APPLY_3]] step %[[CONSTANT_5]] {
// CHECK-NEXT:         %[[APPLY_4:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_1]], %[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_strip_mine_loop_nest(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[RESERVE_SIZE_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[RESERVE_SIZE_0]] : index
// CHECK-NEXT:     %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG0]], %[[MULI_0]]]
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[APPLY_0]] step %[[CONSTANT_3]] {
// CHECK-NEXT:       %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[APPLY_1]], %[[MULI_0]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ARG0]], %[[ADDI_0]] : index
// CHECK-NEXT:       %[[APPLY_2:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       %[[APPLY_3:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_0]]){{\[}}%[[MINSI_0]], %[[MULI_0]]]
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[APPLY_3]] step %[[CONSTANT_5]] {
// CHECK-NEXT:         %[[APPLY_4:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_1]], %[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:         %[[ADDI_1:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : index
// CHECK-NEXT:         scf.for %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[ARG1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_sibling_loops_strip_mine(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[RESERVE_SIZE_0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[RESERVE_SIZE_0]] : index
// CHECK-NEXT:     %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG0]], %[[MULI_0]]]
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[APPLY_0]] step %[[CONSTANT_3]] {
// CHECK-NEXT:       %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[APPLY_1]], %[[MULI_0]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ARG0]], %[[ADDI_0]] : index
// CHECK-NEXT:       %[[APPLY_2:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       %[[APPLY_3:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_0]]){{\[}}%[[MINSI_0]], %[[MULI_0]]]
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[APPLY_3]] step %[[CONSTANT_5]] {
// CHECK-NEXT:         %[[APPLY_4:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_1]], %[[VAL_0]]){{\[}}%[[MULI_0]]]
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[RESERVE_SIZE_1:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[CONSTANT_1]], %[[RESERVE_SIZE_1]] : index
// CHECK-NEXT:     %[[APPLY_5:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG1]], %[[MULI_1]]]
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[VAL_2:.*]] = %[[CONSTANT_6]] to %[[APPLY_5]] step %[[CONSTANT_7]] {
// CHECK-NEXT:       %[[APPLY_6:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_2]]){{\[}}%[[MULI_1]]]
// CHECK-NEXT:       %[[ADDI_1:.*]] = arith.addi %[[APPLY_6]], %[[MULI_1]] : index
// CHECK-NEXT:       %[[MINSI_1:.*]] = arith.minsi %[[ARG1]], %[[ADDI_1]] : index
// CHECK-NEXT:       %[[APPLY_7:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_2]]){{\[}}%[[MULI_1]]]
// CHECK-NEXT:       %[[APPLY_8:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_2]]){{\[}}%[[MINSI_1]], %[[MULI_1]]]
// CHECK-NEXT:       %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK-NEXT:       scf.for %[[VAL_3:.*]] = %[[CONSTANT_8]] to %[[APPLY_8]] step %[[CONSTANT_9]] {
// CHECK-NEXT:         %[[APPLY_9:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_3]], %[[VAL_2]]){{\[}}%[[MULI_1]]]
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }







func.func @test_strip_mine_with_symbolic_tile(%N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // Create a tile size placeholder
  
  scf.for %i = %c0 to %N step %c1 {
    // Loop body
  }
  
  return
}

func.func @test_strip_mine_loop_nest(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // Create a tile size placeholder
  // Strip mine the outermost loop only
  
  scf.for %i = %c0 to %N step %c1 {
    %tmp = arith.addi %N, %M : index // imperfectly nested
    scf.for %j = %c0 to %M step %c1 {
      // Loop body
    }
  }
  
  return
}


func.func @test_sibling_loops_strip_mine(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // Create tile size placeholders - each loop will use the first available one
  
  // First loop uses tile1
  scf.for %i = %c0 to %N step %c1 {
    // Inner loop body
  }
  
  scf.for %j = %c0 to %M step %c1 {
    // Inner loop body
  }
  
  return
}
