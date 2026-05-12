// RUN: scheduler %s | scheduler | FileCheck %s

// CHECK-LABEL: func.func @test_create_tile_size
func.func @test_create_tile_size() {
  // CHECK: %[[TILE_SIZE:.*]] = ktdf.create_tile_size {divisibility = 4 : index, min_value = 8 : index} : index
  %tile_size = ktdf.create_tile_size {min_value = 8 : index, divisibility = 4 : index} : index
  
  // CHECK: %[[TILE_SIZE2:.*]] = ktdf.create_tile_size {divisibility = 16 : index, min_value = 16 : index} : index
  %tile_size2 = ktdf.create_tile_size {min_value = 16 : index, divisibility = 16 : index} : index
  
  return
}

// CHECK-LABEL: func.func @test_tile_size_in_loop
func.func @test_tile_size_in_loop(%N: index) {
  %c0 = arith.constant 0 : index
  
  // CHECK: %[[TILE:.*]] = ktdf.create_tile_size {divisibility = 1 : index, min_value = 1 : index} : index
  %tile = ktdf.create_tile_size {min_value = 1 : index, divisibility = 1 : index} : index
  
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[TILE]]
  scf.for %i = %c0 to %N step %tile {
    // Loop body placeholder
  }
  
  return
}
