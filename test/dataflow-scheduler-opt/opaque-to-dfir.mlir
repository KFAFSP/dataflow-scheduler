// RUN: dataflow-scheduler-opt -ktir-to-dfir "%s" | FileCheck "%s"

// CHECK-LABEL: module @local_schedule_0 {
// CHECK-LABEL: func.func @local_schedule_0(

// CHECK-DAG:   %[[C0:.+]] = arith.constant 0
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128
// CHECK-DAG:   %[[C0L1LU:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// CHECK-DAG:   %[[C1L1LU:.+]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu", type = "l1lu"} : index
// CHECK-DAG:   %[[C0SFU:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
// CHECK-DAG:   %[[C1SFU:.+]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu", type = "sfu"} : index
// CHECK-DAG:   %[[C0L1SU:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
// CHECK-DAG:   %[[C1L1SU:.+]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1su", type = "l1su"} : index
// CHECK-DAG:   %[[SFUREG:.+]] = dataflow.get_unit {name = "sfu_reg", type = "sfu_reg"} : index

// CHECK:       dataflow.program_unit iter_arg

// CHECK:       dataflow.program_unit iter_arg : %[[ARG1:.+]] -> (%[[C0SFU]], %[[C1SFU]]) : {
// CHECK:         %[[MAP0:.+]] = uniform.def_immutable_mapping([%[[C0SFU]] -> %[[C0L1LU]]], [%[[C1SFU]] -> %[[C1L1LU]]]):index
// CHECK:         %[[FROM:.+]] = uniform.query_map(map:%[[MAP0]], key:%[[ARG1]]) : index
// CHECK:         %[[RECV:.+]] = dataflow.receive %[[FROM]] : vector<64xf16>
// CHECK:         %[[R0:.+]] = dataflow.get_logical_memory_view %[[SFUREG]], %[[C0]] {layout_map = #map3} : index, index, memref<64xf16>
// CHECK:         %[[CST:.+]] = vectorchain.constant_bitstream {value = [0x3c00]} : vector<1xf16>
// CHECK:         %[[SPLAT:.+]] = vectorchain.shuffle input(%[[CST]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK:         agen.vector_store %[[SPLAT]], %[[R0]][%[[C0]]] {store_order = #map3, store_set = #set2} : memref<64xf16>, vector<64xf16>
// CHECK:         %[[R1:.+]] = dataflow.get_logical_memory_view %[[SFUREG]], %[[C64]] {layout_map = #map3} : index, index, memref<64xf16>
// CHECK:         %[[R2:.+]] = dataflow.get_logical_memory_view %[[SFUREG]], %[[C128]] {layout_map = #map3} : index, index, memref<64xf16>
// CHECK:         agen.vector_store %[[RECV]], %[[R1]][%[[C0]]] {store_order = #map3, store_set = #set2} : memref<64xf16>, vector<64xf16>
// CHECK:         dataflow.opaque {dataflow_scheduler.register_names = ["c0", "in0", "out0", "t0_0"], func_name = "fake_exp", parameter_dictionary = {}, read_only_register_dictionary = {c0 = "R0", in0 = "R1"}, read_write_register_dictionary = {out0 = "R2", t0_0 = "R3"}}
// CHECK:         %[[RET:.+]] = agen.vector_load %[[R2]][%[[C0]]] {load_order = #map3, load_set = #set2} : memref<64xf16>, vector<64xf16>
// CHECK:         %[[MAP1:.+]] = uniform.def_immutable_mapping([%[[C0SFU]] -> %[[C0L1SU]]], [%[[C1SFU]] -> %[[C1L1SU]]]):index
// CHECK:         %[[TO:.+]] = uniform.query_map(map:%[[MAP1]], key:%[[ARG1]]) : index
// CHECK:         dataflow.send %[[TO]], %[[RET]] : vector<64xf16>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @arith_math_test(%i: index) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct memory views
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %start_row_i = arith.addi %start_row, %i : index
    
    // Construct access tiles
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

    // Load data
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

    %result_empty = tensor.empty() : tensor<1x64xf16>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ], 
      iterator_types = ["parallel", "parallel"]
    }
      ins(%A_data_tile : tensor<1x64xf16>)
      outs(%result_empty: tensor<1x64xf16>)
    {
      ^bb0(%arg0: f16, %arg1: f16):
        %0 = spyreop.exp %arg0 : f16
        linalg.yield %0 : f16
    } -> tensor<1x64xf16>

    // Construct access tile for output
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row_i, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

    // Store result
    ktdp.store %result, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    return
  }
}
