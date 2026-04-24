// RUN: scheduler %s | scheduler | FileCheck %s

// Test LoopType attribute on scf.for loops using pretty-printed syntax

// CHECK-LABEL: func.func @test_loop_type_attr
func.func @test_loop_type_attr() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: scf.for {{.*}} {
  // CHECK-NEXT: } {loop_type = #ktdf.loop_type<parallel_loop>}
  scf.for %i = %c0 to %c10 step %c1 {
    // Parallel loop body
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  
  // CHECK: scf.for {{.*}} {
  // CHECK-NEXT: } {loop_type = #ktdf.loop_type<reduction_loop>}
  scf.for %j = %c0 to %c10 step %c1 {
    // Reduction loop body
  } {loop_type = #ktdf.loop_type<reduction_loop>}
  
  return
}