// RUN: scheduler --strip-mine-scf-for-loops="strip-mine-size=-1" %s | FileCheck %s

// Test that negative strip-mine size causes the pass to use ktdf.create_tile_size operations

// CHECK-LABEL: func.func @test_strip_mine_with_symbolic_tile
func.func @test_strip_mine_with_symbolic_tile(%N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // Create a tile size placeholder
  // CHECK: %[[TILE:.*]] = ktdf.create_tile_size {divisibility = 128 : index, min_value = 16 : index}
  %tile = ktdf.create_tile_size {min_value = 16 : index, divisibility = 128 : index} : index
  
  // CHECK: %[[MUL1:.*]] = arith.muli %{{.*}}, %[[TILE]]
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[MUL1]]
  // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
  scf.for %i = %c0 to %N step %c1 {
    // Loop body
  }
  
  return
}

// CHECK-LABEL: func.func @test_multiple_loops_strip_mine
func.func @test_multiple_loops_strip_mine(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // Create tile size placeholders - each loop will use the first available one
  // CHECK: %[[TILE1:.*]] = ktdf.create_tile_size {divisibility = 64 : index, min_value = 8 : index}
  %tile1 = ktdf.create_tile_size {min_value = 8 : index, divisibility = 64 : index} : index
  
  // CHECK: %[[TILE2:.*]] = ktdf.create_tile_size {divisibility = 32 : index, min_value = 4 : index}
  %tile2 = ktdf.create_tile_size {min_value = 4 : index, divisibility = 32 : index} : index
  
  // First loop uses tile1
  // CHECK: %[[MUL1:.*]] = arith.muli %{{.*}}, %[[TILE1]]
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[MUL1]]
  scf.for %i = %c0 to %N step %c1 {
    // Inner loop body
  }
  
  // Second loop uses tile2 (since tile1 was already used)
  // Note: In the current implementation, each loop independently searches for the first available tile
  // So both loops will actually use tile1. This is a limitation of the current design.
  // CHECK: %[[MUL2:.*]] = arith.muli %{{.*}}, %[[TILE1]]
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[MUL2]]
  scf.for %j = %c0 to %M step %c1 {
    // Inner loop body
  }
  
  return
}