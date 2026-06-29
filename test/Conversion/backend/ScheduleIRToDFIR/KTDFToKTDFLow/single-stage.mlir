// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   func.func @single_stage_pipeline(
// CHECK-SAME:      %[[ARG0:.*]]: index) attributes {grid = [2]} {
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[CONSTANT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[CONSTANT_7:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_1:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_1]], %[[CONSTANT_6]] : index
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG0]] : index
// CHECK-NEXT:     %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_10:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_11:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_7]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[CONSTANT_12:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_12]] : index
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[CONSTRUCT_MEMORY_VIEW_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_8]] to %[[CONSTANT_9]] step %[[CONSTANT_4]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_10]] to %[[CONSTANT_11]] step %[[CONSTANT_5]] {
// CHECK-NEXT:         ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] to %[[ALLOC_0]]{{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }







// Single-stage pipeline test
// Tests: basic transformation with no signal insertion
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @single_stage_pipeline(%arg0: index) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %c1024 = arith.constant 1024 : index

    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.muli %0, %c3 : index
    %2 = arith.addi %1, %arg0 : index
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %c64_3 = arith.constant 64 : index

    %3 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, "DDR">
    %c64_4 = arith.constant 64 : index
    %4 = arith.muli %2, %c64_4 : index
    %reinterpret_cast = memref.reinterpret_cast %3 to offset: [%4], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">

    scf.for %arg1 = %c0_1 to %c1_2 step %c1 {
      scf.for %arg2 = %c0_2 to %c64_3 step %c64 {
        ktdf.pipeline {
          %5:2 = ktdf.private -> (memref<1x64xf16, "L1">, !ktdf.token) {
            %alloc = memref.alloc() : memref<1x64xf16, "L1">
            %token = ktdf.create_token : !ktdf.token
            ktdf.private_yield %alloc, %token : memref<1x64xf16, "L1">, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%5#1) {
            ktdf.data_transfer from %reinterpret_cast[%arg1, %arg2] size [1, 64] to %5#0[%c0, %c0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
          } {applicable_units = ["MNILU"]}
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

