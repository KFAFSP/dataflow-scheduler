// RUN: dataflow-scheduler-opt --affine-min-canonicalization -allow-unregistered-dialect %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, 2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, 4)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL:   func.func @test_basic_canonicalization(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[MIN_0:.*]] = affine.min #[[$ATTR_0]](%[[VAL_0]]){{\[}}%[[CONSTANT_2]], %[[CONSTANT_1]]]
// CHECK-NEXT:       "test.use"(%[[MIN_0]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_with_different_constants(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 100 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[MIN_0:.*]] = affine.min #[[$ATTR_1]](%[[VAL_0]]){{\[}}%[[CONSTANT_2]], %[[CONSTANT_1]]]
// CHECK-NEXT:       "test.use"(%[[MIN_0]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_reversed_minsi_operands(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[MIN_0:.*]] = affine.min #[[$ATTR_0]](%[[VAL_0]]){{\[}}%[[CONSTANT_2]], %[[CONSTANT_1]]]
// CHECK-NEXT:       "test.use"(%[[MIN_0]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_reversed_addi_operands(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[MIN_0:.*]] = affine.min #[[$ATTR_0]](%[[VAL_0]]){{\[}}%[[CONSTANT_2]], %[[CONSTANT_1]]]
// CHECK-NEXT:       "test.use"(%[[MIN_0]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_with_multiple_uses(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:       %[[MIN_0:.*]] = affine.min #[[$ATTR_0]](%[[VAL_0]]){{\[}}%[[CONSTANT_2]], %[[CONSTANT_1]]]
// CHECK-NEXT:       "test.use"(%[[MIN_0]]) : (index) -> ()
// CHECK-NEXT:       "test.use"(%[[MINSI_0]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_negative_wrong_map(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_2]] : index
// CHECK-NEXT:       %[[APPLY_0:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[VAL_0]], %[[MINSI_0]]]
// CHECK-NEXT:       "test.use"(%[[APPLY_0]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Test basic canonicalization of arith.addi + arith.minsi + affine.apply to affine.min

#map = affine_map<()[s0, s1] -> (-s0 + s1)>

func.func @test_basic_canonicalization(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  
  scf.for %iv = %c0 to %c8 step %c2 {
    %add = arith.addi %iv, %c2 : index
    %min = arith.minsi %add, %c8 : index
    %result = affine.apply #map()[%iv, %min]
    
    // Use the result to prevent DCE
    "test.use"(%result) : (index) -> ()
  }
  return
}

func.func @test_with_different_constants(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : index
  
  scf.for %iv = %c0 to %c100 step %c4 {
    %add = arith.addi %iv, %c4 : index
    %min = arith.minsi %add, %c100 : index
    %result = affine.apply #map()[%iv, %min]
    
    "test.use"(%result) : (index) -> ()
  }
  return
}

func.func @test_reversed_minsi_operands(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  
  scf.for %iv = %c0 to %c8 step %c2 {
    // Test with minsi operands reversed (ub, add) instead of (add, ub)
    %add = arith.addi %iv, %c2 : index
    %min = arith.minsi %c8, %add : index
    %result = affine.apply #map()[%iv, %min]
    
    "test.use"(%result) : (index) -> ()
  }
  return
}

func.func @test_reversed_addi_operands(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  
  scf.for %iv = %c0 to %c8 step %c2 {
    // Test with addi operands reversed (step, iv) instead of (iv, step)
    %add = arith.addi %c2, %iv : index
    %min = arith.minsi %add, %c8 : index
    %result = affine.apply #map()[%iv, %min]
    
    "test.use"(%result) : (index) -> ()
  }
  return
}

func.func @test_with_multiple_uses(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  
  scf.for %iv = %c0 to %c8 step %c2 {
    %add = arith.addi %iv, %c2 : index
    %min = arith.minsi %add, %c8 : index
    %result = affine.apply #map()[%iv, %min]
    
    "test.use"(%result) : (index) -> ()
    "test.use"(%min) : (index) -> ()  // Extra use of %min
  }
  return
}

// Negative test: pattern should NOT match if affine.apply has wrong map
#map_wrong = affine_map<()[s0, s1] -> (s0 + s1)>

func.func @test_negative_wrong_map(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  
  scf.for %iv = %c0 to %c8 step %c2 {
    %add = arith.addi %iv, %c2 : index
    %min = arith.minsi %add, %c8 : index
    %result = affine.apply #map_wrong()[%iv, %min]
    
    "test.use"(%result) : (index) -> ()
  }
  return
}
