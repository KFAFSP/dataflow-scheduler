// RUN: dataflow-scheduler-opt --allow-unregistered-dialect --ktir-map-and-tile %s \
// RUN:   | FileCheck %s

// Tests that stored data tiles with other users are tiled without changing what
// those users observe, and that stores only move into loop nests if their
// access tiles can be computed in front of them.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#set3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // A user after the store reads the stored data back from memory.
  // CHECK-LABEL: func.func @used_after_store
  // CHECK:         ktdp.construct_access_tile
  // CHECK:         %[[TILE_OUT:.+]] = ktdp.construct_access_tile
  // CHECK-NOT:     iter_args
  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             scf.for
  // CHECK:               %[[EXP:.+]] = linalg.generic
  // CHECK:               ktdp_lowering.store %[[EXP]] into %[[TILE_OUT]]
  // CHECK:         %[[RELOAD:.+]] = ktdp.load %[[TILE_OUT]] : <6x64x64xindex> -> tensor<6x64x64xf16>
  // CHECK-NEXT:    "test.use"(%[[RELOAD]]) : (tensor<6x64x64xf16>) -> ()
  func.func @used_after_store() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c49152 = arith.constant 49152 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    %2 = ktdp.load %1 : <6x64x64xindex> -> tensor<6x64x64xf16>
    %3 = tensor.empty() : tensor<6x64x64xf16>
    %4 = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%2 : tensor<6x64x64xf16>) outs(%3 : tensor<6x64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %e = spyreop.exp %in : f16
      linalg.yield %e : f16
    } -> tensor<6x64x64xf16>
    %5 = ktdp.construct_memory_view %c49152, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    ktdp.store %4, %6 : tensor<6x64x64xf16>, <6x64x64xindex>
    "test.use"(%4) : (tensor<6x64x64xf16>) -> ()
    return
  }

  // The reloaded data is reshaped to the type that the user observed.
  // CHECK-LABEL: func.func @used_after_store_reshaped
  // CHECK:         ktdp.construct_access_tile
  // CHECK:         %[[TILE_OUT:.+]] = ktdp.construct_access_tile
  // CHECK-NOT:     iter_args
  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             %[[EXP:.+]] = linalg.generic
  // CHECK:             %[[EXPANDED:.+]] = tensor.expand_shape %[[EXP]]
  // CHECK-NEXT:        ktdp_lowering.store %[[EXPANDED]] into %[[TILE_OUT]]
  // CHECK:         %[[RELOAD:.+]] = ktdp.load %[[TILE_OUT]] : <1x64x64xindex> -> tensor<1x64x64xf16>
  // CHECK-NEXT:    %[[COLLAPSED:.+]] = tensor.collapse_shape %[[RELOAD]]
  // CHECK-SAME:      tensor<1x64x64xf16> into tensor<64x64xf16>
  // CHECK-NEXT:    "test.use"(%[[COLLAPSED]]) : (tensor<64x64xf16>) -> ()
  func.func @used_after_store_reshaped() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c49152 = arith.constant 49152 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0] {access_tile_order = #id2, access_tile_set = #set2} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
    %2 = ktdp.load %1 : <64x64xindex> -> tensor<64x64xf16>
    %3 = tensor.empty() : tensor<64x64xf16>
    %4 = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
         ins(%2 : tensor<64x64xf16>) outs(%3 : tensor<64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %e = spyreop.exp %in : f16
      linalg.yield %e : f16
    } -> tensor<64x64xf16>
    %expanded = tensor.expand_shape %4 [[0, 1], [2]] output_shape [1, 64, 64] : tensor<64x64xf16> into tensor<1x64x64xf16>
    %5 = ktdp.construct_memory_view %c49152, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16>
    %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set1} : memref<1x64x64xf16> -> !ktdp.access_tile<1x64x64xindex>
    ktdp.store %expanded, %6 : tensor<1x64x64xf16>, <1x64x64xindex>
    "test.use"(%4) : (tensor<64x64xf16>) -> ()
    return
  }

  // A pure user before the store also reads the data back after the loop nest,
  // in front of which the access tile is computed.
  // CHECK-LABEL: func.func @pure_use_before_store
  // CHECK:         ktdp.construct_access_tile
  // CHECK:         %[[TILE_OUT:.+]] = ktdp.construct_access_tile
  // CHECK-NOT:     iter_args
  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             scf.for
  // CHECK:               %[[EXP:.+]] = linalg.generic
  // CHECK:               ktdp_lowering.store %[[EXP]] into %[[TILE_OUT]]
  // CHECK:         %[[RELOAD:.+]] = ktdp.load %[[TILE_OUT]] : <6x64x64xindex> -> tensor<6x64x64xf16>
  // CHECK-NEXT:    %[[ELEM:.+]] = tensor.extract %[[RELOAD]]
  // CHECK-NOT:     ktdp.store
  // CHECK:         "test.use"(%[[ELEM]]) : (f16) -> ()
  func.func @pure_use_before_store() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c49152 = arith.constant 49152 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    %2 = ktdp.load %1 : <6x64x64xindex> -> tensor<6x64x64xf16>
    %3 = tensor.empty() : tensor<6x64x64xf16>
    %4 = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%2 : tensor<6x64x64xf16>) outs(%3 : tensor<6x64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %e = spyreop.exp %in : f16
      linalg.yield %e : f16
    } -> tensor<6x64x64xf16>
    %x = tensor.extract %4[%c0, %c0, %c0] : tensor<6x64x64xf16>
    %5 = ktdp.construct_memory_view %c49152, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    ktdp.store %4, %6 : tensor<6x64x64xf16>, <6x64x64xindex>
    "test.use"(%x) : (f16) -> ()
    return
  }
  // Memory effects between the loop nest and the store prevent moving the store
  // into the loop nest.
  // CHECK-LABEL: func.func @effect_before_store
  // CHECK:         %[[NEST:.+]] = scf.for {{.+}} iter_args
  // CHECK:               tensor.insert_slice
  // CHECK:         "test.use"(%[[NEST]])
  // CHECK:         ktdp.store %[[NEST]]
  func.func @effect_before_store() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c49152 = arith.constant 49152 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    %2 = ktdp.load %1 : <6x64x64xindex> -> tensor<6x64x64xf16>
    %3 = tensor.empty() : tensor<6x64x64xf16>
    %4 = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%2 : tensor<6x64x64xf16>) outs(%3 : tensor<6x64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %e = spyreop.exp %in : f16
      linalg.yield %e : f16
    } -> tensor<6x64x64xf16>
    "test.use"(%4) : (tensor<6x64x64xf16>) -> ()
    %5 = ktdp.construct_memory_view %c49152, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    ktdp.store %4, %6 : tensor<6x64x64xf16>, <6x64x64xindex>
    return
  }
  // The store can not move into the loop nest if its access tile can not be
  // computed in front of it.
  // CHECK-LABEL: func.func @immovable_access_tile
  // CHECK:         %[[NEST:.+]] = scf.for {{.+}} iter_args
  // CHECK:               tensor.insert_slice
  // CHECK:         %[[OFFSET:.+]] = "test.offset"()
  // CHECK:         %[[TILE_OUT:.+]] = ktdp.construct_access_tile {{.+}}[%[[OFFSET]],
  // CHECK:         ktdp.store %[[NEST]], %[[TILE_OUT]]
  func.func @immovable_access_tile() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c49152 = arith.constant 49152 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    %2 = ktdp.load %1 : <6x64x64xindex> -> tensor<6x64x64xf16>
    %3 = tensor.empty() : tensor<6x64x64xf16>
    %4 = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%2 : tensor<6x64x64xf16>) outs(%3 : tensor<6x64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %e = spyreop.exp %in : f16
      linalg.yield %e : f16
    } -> tensor<6x64x64xf16>
    %5 = ktdp.construct_memory_view %c49152, sizes: [6, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<6x64x64xf16>
    %offset = "test.offset"() : () -> index
    %6 = ktdp.construct_access_tile %5[%offset, %c0, %c0] {access_tile_order = #id3, access_tile_set = #set3} : memref<6x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>
    ktdp.store %4, %6 : tensor<6x64x64xf16>, <6x64x64xindex>
    return
  }
}
