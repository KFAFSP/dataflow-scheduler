// RUN: scheduler --tile-scf-for-loops="tile-sizes=-1,-1" %s | FileCheck %s

// Test that negative tile sizes cause the pass to use ktdf.create_tile_size operations

// CHECK-LABEL: func.func @test_two_loops_two_tiles
func.func @test_two_loops_two_tiles(%N: index, %M: index) {
  %c1 = arith.constant 1 : index
  
  // Create two tile size placeholders
  // CHECK: %[[TILE1:.*]] = ktdf.create_tile_size {divisibility = 1 : index, min_value = 8 : index}
  %tile1 = ktdf.create_tile_size {min_value = 8 : index, divisibility = 1 : index} : index
  
  // CHECK: %[[TILE2:.*]] = ktdf.create_tile_size {divisibility = 1 : index, min_value = 16 : index}
  %tile2 = ktdf.create_tile_size {min_value = 16 : index, divisibility = 1 : index} : index
  
  // CHECK: %[[MUL1:.*]] = arith.muli %{{.*}}, %[[TILE1]]
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[MUL1]]
  scf.for %i = %c1 to %N step %c1 {
    // CHECK: %[[MUL2:.*]] = arith.muli %{{.*}}, %[[TILE2]]
    // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[MUL2]]
    scf.for %j = %c1 to %M step %c1 {
      // Loop body
    }
  }
  
  return
}

// CHECK-LABEL: func.func @test_single_loop_single_tile
func.func @test_single_loop_single_tile(%N: index) {
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[TILE:.*]] = ktdf.create_tile_size {divisibility = 1 : index, min_value = 1 : index}
  %tile = ktdf.create_tile_size {min_value = 1 : index, divisibility = 1 : index} : index
  
  // CHECK: %[[MUL:.*]] = arith.muli %{{.*}}, %[[TILE]]
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[MUL]]
  scf.for %i = %c1 to %N step %c1 {
    // Loop body
  }
  
  return
}