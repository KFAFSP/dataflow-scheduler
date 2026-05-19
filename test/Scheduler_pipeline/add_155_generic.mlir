// RUN: scheduler --compute-group-extraction %s | FileCheck %s

// After compute-group extraction, the original ops in @Add_155 have been
// cloned into a child module. The leftover pure ops in the top-level function
// must be removed by the in-pass runRegionDCE call, leaving only the call to
// the extracted local-schedule and the return.
//
// CHECK-LABEL: func.func @Add_155()
// CHECK-NEXT:    call @"local-schedule-0"() : () -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  func.func @Add_155() {
    %c64000 = arith.constant 64000 : index
    %0 = ktdp.construct_memory_view %c64000, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x64x64xf16>
    %c0 = arith.constant 0 : index
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<12x64x64xf16> -> !ktdp.access_tile<12x64x64xindex>
    %c113152 = arith.constant 113152 : index
    %2 = ktdp.construct_memory_view %c113152, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x64x64xf16>
    %c0_0 = arith.constant 0 : index
    %3 = ktdp.construct_access_tile %2[%c0_0, %c0_0, %c0_0] {access_tile_order = #map, access_tile_set = #set} : memref<12x64x64xf16> -> !ktdp.access_tile<12x64x64xindex>
    %c162304 = arith.constant 162304 : index
    %4 = ktdp.construct_memory_view %c162304, sizes: [1, 12, 64, 64], strides: [49152, 4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1x12x64x64xf16>
    %c0_1 = arith.constant 0 : index
    %5 = ktdp.construct_access_tile %4[%c0_1, %c0_1, %c0_1, %c0_1] {access_tile_order = #map1, access_tile_set = #set1} : memref<1x12x64x64xf16> -> !ktdp.access_tile<1x12x64x64xindex>
    %6 = ktdp.load %1 : <12x64x64xindex> -> tensor<12x64x64xf16>
    %7 = ktdp.load %3 : <12x64x64xindex> -> tensor<12x64x64xf16>
    %8 = tensor.empty() : tensor<1x12x64x64xf16>
    %9 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %7 : tensor<12x64x64xf16>, tensor<12x64x64xf16>) outs(%8 : tensor<1x12x64x64xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %10 = arith.addf %in, %in_2 : f16
      linalg.yield %10 : f16
    } -> tensor<1x12x64x64xf16>
    ktdp.store %9, %5 : tensor<1x12x64x64xf16>, <1x12x64x64xindex>
    return
  }
}
