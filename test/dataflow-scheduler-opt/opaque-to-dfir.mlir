// RUN: dataflow-scheduler-opt -ktir-to-dfir "%s" | FileCheck "%s"

// CHECK-LABEL: module @local_schedule_0 {
// CHECK-LABEL: func.func @local_schedule_0(

// TODO: Match this
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128
// %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
// %1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// %3 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu-CL0", type = "l1lu"} : index
// %5 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu-CL0", type = "l1lu"} : index
// %6 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-l1lu-CL1", type = "l1lu"} : index
// %7 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-l1lu-CL1", type = "l1lu"} : index
// %8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu-CL0", type = "sfu"} : index
// %9 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu-CL0", type = "sfu"} : index
// %10 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfu-CL1", type = "sfu"} : index
// %11 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfu-CL1", type = "sfu"} : index
// CHECK-DAG:   %[[C0SFU0:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu-CL0", type = "sfu"} : index
// CHECK-DAG:   %[[C1SFU0:.+]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu-CL0", type = "sfu"} : index
// CHECK-DAG:   %[[C0SFU1:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfu-CL1", type = "sfu"} : index
// CHECK-DAG:   %[[C1SFU1:.+]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfu-CL1", type = "sfu"} : index
// %16 = dataflow.get_unit {name = "sfu_reg", type = "sfu_reg"} : index
// %17 = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// %18 = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// %19 = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// dataflow.program_unit iter_arg : %arg0 -> (%8, %9, %10, %11) : {
//   %20 = dataflow.get_logical_memory_view %16, %c0 {layout_map = #map} : index, index, memref<64xf16>
//   %21 = vectorchain.constant_bitstream {value = [0x3c00]} : vector<1xf16>
//   %22 = vectorchain.shuffle input(%21) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
//   agen.vector_store %22, %20[%c0] {store_order = #map, store_set = #set} : memref<64xf16>, vector<64xf16>
//   scf.for %arg1 = %c0 to %c3 step %c1 {
//     scf.for %arg2 = %c0 to %c32 step %c1 {
//       scf.for %arg3 = %c0 to %c2 step %c1 {
//         %23 = uniform.def_immutable_mapping([%8 -> %4], [%9 -> %5], [%10 -> %6], [%11 -> %7]):index
//         %24 = uniform.query_map(map:%23, key:%arg0) : index
//         %25 = dataflow.receive %24 : vector<64xf16>
//         %26 = dataflow.get_logical_memory_view %16, %c64 {layout_map = #map} : index, index, memref<64xf16>
//         %27 = dataflow.get_logical_memory_view %16, %c128 {layout_map = #map} : index, index, memref<64xf16>
//         agen.vector_store %25, %26[%c0] {store_order = #map, store_set = #set} : memref<64xf16>, vector<64xf16>
//         dataflow.opaque {dataflow_scheduler.register_names = ["c0", "in0", "out0", "t0_0"], func_name = "fake_exp", parameter_dictionary = {}, read_only_register_dictionary = {c0 = "R0", in0 = "R1"}, read_write_register_dictionary = {out0 = "R2", t0_0 = "R3"}}
//         %28 = agen.vector_load %27[%c0] {load_order = #map, load_set = #set} : memref<64xf16>, vector<64xf16>
//         %29 = uniform.def_immutable_mapping([%8 -> %12], [%9 -> %13], [%10 -> %14], [%11 -> %15]):index
//         %30 = uniform.query_map(map:%29, key:%arg0) : index
//         dataflow.send %30, %28 : vector<64xf16>
//       }
//     }
//   }
// }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_whole = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_out = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 5 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @Exp_1(%base_in: index, %base_out: index) attributes {grid = [2]} {
    %zero = arith.constant 0 : index
    %tile = ktdp.get_compute_tile_id : index

    %view_in = ktdp.construct_memory_view %base_in, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set_whole, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf16>
    %tile_in = ktdp.construct_access_tile %view_in[%tile * 6, %zero, %zero] {access_tile_order = #map, access_tile_set = #set} : memref<12x64x64xf16> -> !ktdp.access_tile<6x64x64xindex>

    %view_out = ktdp.construct_memory_view %base_out, sizes: [1, 12, 64, 64], strides: [49152, 4096, 64, 1] {coordinate_set = #set_out, memory_space = #ktdp.memory_space<global>} : memref<1x12x64x64xf16>
    %tile_out = ktdp.construct_access_tile %view_out[%zero, %tile * 6, %zero, %zero] {access_tile_order = #map1, access_tile_set = #set1} : memref<1x12x64x64xf16> -> !ktdp.access_tile<1x6x64x64xindex>

    %in = ktdp.load %tile_in : <6x64x64xindex> -> tensor<6x64x64xf16>
    %init = tensor.empty() : tensor<1x6x64x64xf16>
    %result = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : tensor<6x64x64xf16>) outs(%init : tensor<1x6x64x64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %e = spyreop.exp %x : f16
      linalg.yield %e : f16
    } -> tensor<1x6x64x64xf16>
    ktdp.store %result, %tile_out : tensor<1x6x64x64xf16>, <1x6x64x64xindex>
    return
  }
}
