// RUN: dataflow-scheduler-opt --address-assignment %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @address_assignment_basic(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[ARG0]] : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[MULI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:     %[[MULI_2:.*]] = arith.muli %[[MULI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[MULI_2]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]] = ktdf.private -> (memref<1x64xf16, "L1">) {
// CHECK-NEXT:         %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_4]] : index to memref<1x64xf16, "L1">
// CHECK-NEXT:         ktdf.private_yield %[[UNREALIZED_CONVERSION_CAST_0]] : memref<1x64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:         ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 64] to %[[PRIVATE_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:         ktdf.data_transfer from %[[PRIVATE_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 64] to %[[REINTERPRET_CAST_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size [1, 64] : memref<1x64xf16, "L1">, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       } {applicable_units = ["MNILU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



#map = affine_map<(d0) -> (d0 * 64)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  
  func.func @address_assignment_basic(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c12288 = arith.constant 12288 : index
    
    %tile_id = ktdp.get_compute_tile_id : index
    %mul = arith.muli %tile_id, %arg0 : index
    
    // Input memory view
    %input_view = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>
    %input_offset = arith.muli %mul, %c64 : index
    %input_cast = memref.memory_space_cast %input_view : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
    %input = memref.reinterpret_cast %input_cast to offset: [%input_offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    
    // Output memory view
    %output_view = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>>
    %output_offset = arith.muli %mul, %c64 : index
    %output_cast = memref.memory_space_cast %output_view : memref<96x64xf16, #ktdp.spyre_memory_space<HBM>> to memref<96x64xf16, "DDR">
    %output = memref.reinterpret_cast %output_cast to offset: [%output_offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    
    ktdf.pipeline {
      %buffer = ktdf.private -> (memref<1x64xf16, "L1">) {
        %alloc = memref.alloc() : memref<1x64xf16, "L1">
        ktdf.private_yield %alloc : memref<1x64xf16, "L1">
      }
      
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.data_transfer from %input[%c0, %c0] size [1, 64] to %buffer[%c0, %c0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        ktdf.data_transfer from %buffer[%c0, %c0] size [1, 64] to %output[%c0, %c0] size [1, 64] : memref<1x64xf16, "L1">, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      } {applicable_units = ["MNILU"]}
    }
    
    return
  }
}
