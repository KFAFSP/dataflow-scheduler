// RUN: scheduler --construct-three-stage-pipeline %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[MULI_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:     %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[MULI_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:     %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <1x64xindex> -> tensor<1x64xf16>
// CHECK-NEXT:     %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <1x64xindex> -> tensor<1x64xf16>
// CHECK-NEXT:     %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:     %[[ADD_0:.*]] = linalg.add ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) -> tensor<1x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[MULI_0]], %[[CONSTANT_0]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
// CHECK-NEXT:     ktdp.store %[[EMPTY_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<1x64xf16>, <1x64xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/






module {
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %tile_size = arith.constant 3 : index
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %C_start_address = arith.constant 18432 : index
    
        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index
    
        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
        
        // Construct an access tile from the memory view of A
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row+%i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Construct an access tile from the memory view of B
        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row+%i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        // C[start_row+i:start_row+i+2][c0:c0+63] = A[start_row+i:start_row+i+2][c0:c0+63] + B[start_row+i:start_row+i+2][c0:c0+63]
        // Perform add operation on the data tiles.
        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        %result = linalg.add ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
                    outs(%C_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        // Construct an access tile from the memory view of C
        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row+%i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %result, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
        
        return
    }
}