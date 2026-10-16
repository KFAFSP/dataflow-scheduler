// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// Tests that the order of via hops is preserved when tiling loads and stores.
// The memory spaces of the hops are synthetic.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#set3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // CHECK-LABEL: func.func @plain
  // CHECK:         %[[TILE_IN:.+]] = ktdp.construct_access_tile
  // CHECK:         %[[TILE_OUT:.+]] = ktdp.construct_access_tile
  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             scf.for
  // CHECK-NEXT:          %[[LOAD:.+]] = ktdp_lowering.load %[[TILE_IN]]
  // CHECK-NEXT:          %[[HOP_IN:.+]] = ktdf.via["A", "B", "C", "D"] %[[LOAD]]
  // CHECK-NEXT:          %[[INIT:.+]] = tensor.empty() : tensor<1x1x64xf16>
  // CHECK-NEXT:          %[[EXP:.+]] = linalg.generic
  // CHECK-SAME:            ins(%[[HOP_IN]] : tensor<1x1x64xf16>) outs(%[[INIT]] : tensor<1x1x64xf16>)
  // CHECK:               %[[HOP_OUT:.+]] = ktdf.via["D", "C", "B", "A"] %[[EXP]]
  // CHECK-NEXT:          ktdp_lowering.store %[[HOP_OUT]] into %[[TILE_OUT]]
  func.func @plain() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c49152 = arith.constant 49152 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    %2 = ktdp.load %1 : <6x64x64xindex> -> tensor<6x64x64xf16>
    %hop_in = ktdf.via["A", "B", "C", "D"] %2 : tensor<6x64x64xf16>
    %3 = tensor.empty() : tensor<6x64x64xf16>
    %4 = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%hop_in : tensor<6x64x64xf16>) outs(%3 : tensor<6x64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %e = spyreop.exp %in : f16
      linalg.yield %e : f16
    } -> tensor<6x64x64xf16>
    %hop_out = ktdf.via["D", "C", "B", "A"] %4 : tensor<6x64x64xf16>
    %5 = ktdp.construct_memory_view %c49152, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    ktdp.store %hop_out, %6 : tensor<6x64x64xf16>, <6x64x64xindex>
    return
  }

  // The via hops move after the reshapes, which keeps their order.
  // CHECK-LABEL: func.func @reshaped
  // CHECK:         %[[TILE_IN:.+]] = ktdp.construct_access_tile
  // CHECK:         %[[TILE_OUT:.+]] = ktdp.construct_access_tile
  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK-NEXT:        %[[LOAD:.+]] = ktdp_lowering.load %[[TILE_IN]]
  // CHECK-SAME:          -> tensor<1x1x64xf16>
  // CHECK-NEXT:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[LOAD]]
  // CHECK-SAME:          tensor<1x1x64xf16> into tensor<1x64xf16>
  // CHECK-NEXT:        %[[HOP_IN:.+]] = ktdf.via["A", "B", "C", "D"] %[[COLLAPSED]]
  // CHECK-NEXT:        %[[INIT:.+]] = tensor.empty() : tensor<1x64xf16>
  // CHECK-NEXT:        %[[EXP:.+]] = linalg.generic
  // CHECK-SAME:          ins(%[[HOP_IN]] : tensor<1x64xf16>) outs(%[[INIT]] : tensor<1x64xf16>)
  // CHECK:             %[[EXPANDED:.+]] = tensor.expand_shape %[[EXP]]
  // CHECK-SAME:          tensor<1x64xf16> into tensor<1x1x64xf16>
  // CHECK-NEXT:        %[[HOP_OUT:.+]] = ktdf.via["D", "C", "B", "A"] %[[EXPANDED]]
  // CHECK-NEXT:        ktdp_lowering.store %[[HOP_OUT]] into %[[TILE_OUT]]
  func.func @reshaped() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c49152 = arith.constant 49152 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set1} : memref<1x64x64xf16> -> !ktdp.access_tile<1x64x64xindex>
    %2 = ktdp.load %1 : <1x64x64xindex> -> tensor<1x64x64xf16>
    %hop_in = ktdf.via["A", "B", "C", "D"] %2 : tensor<1x64x64xf16>
    %collapsed = tensor.collapse_shape %hop_in [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
    %3 = tensor.empty() : tensor<64x64xf16>
    %4 = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
         ins(%collapsed : tensor<64x64xf16>) outs(%3 : tensor<64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %e = spyreop.exp %in : f16
      linalg.yield %e : f16
    } -> tensor<64x64xf16>
    %expanded = tensor.expand_shape %4 [[0, 1], [2]] output_shape [1, 64, 64] : tensor<64x64xf16> into tensor<1x64x64xf16>
    %hop_out = ktdf.via["D", "C", "B", "A"] %expanded : tensor<1x64x64xf16>
    %5 = ktdp.construct_memory_view %c49152, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16>
    %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set1} : memref<1x64x64xf16> -> !ktdp.access_tile<1x64x64xindex>
    ktdp.store %hop_out, %6 : tensor<1x64x64xf16>, <1x64x64xindex>
    return
  }
}
