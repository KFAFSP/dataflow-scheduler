// RUN: dataflow-scheduler-opt --tile-size-selection --canonicalize %s | FileCheck %s

// CHECK-LABEL:   func.func private @tile_size_selection_test() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 6 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  func.func private @tile_size_selection_test() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    
    // Reserve tile sizes - these should be resolved to concrete values
    %tile_size = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    
    // Compute loop bound using reserved size
    %loop_bound = arith.ceildivui %c12, %tile_size : index
    
    // Allocate buffer with dynamic size - should become static
    %alloc = memref.alloc(%tile_size, %tile_size) : memref<?x?x12xf16, "L1">
    
    scf.for %i = %c0 to %loop_bound step %c1 {
      // Derive actual tile size for this iteration
      %actual_size = ktdf.tiling.derive_size [%i : %tile_size], total_size = %c12 : index
      
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          scf.for %j = %c0 to %actual_size step %c1 {
            // Linearize index using tile size
            %linear_idx = ktdf.tiling.linearize_index [%i : %tile_size], [%j : %c1] : index
            
            // Compute subscripts with arithmetic that should be simplified
            %sub1 = arith.subi %j, %c0 : index
            %div1 = arith.divsi %sub1, %c1 : index
            
            %val = memref.load %alloc[%div1, %div1, %linear_idx] : memref<?x?x12xf16, "L1">
          }
        } {applicable_units = ["MNILU"]}
      }
    }
    
    memref.dealloc %alloc : memref<?x?x12xf16, "L1">
    return
  }
}
