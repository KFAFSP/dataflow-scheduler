// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)"  -allow-unregistered-dialect %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK-LABEL:   func.func @basic_1d_balanced() attributes {grid = [2]} {
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu-CL0", type = "sfu"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu-CL0", type = "sfu"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfu-CL1", type = "sfu"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfu-CL1", type = "sfu"} : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[CONSTANT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_2]] -> %[[GET_UNIT_2]]], {{\[}}%[[CONSTANT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:     %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_6:.*]] = arith.constant 4 : index
// CHECK-NEXT:     ktdf.parallel (%[[VAL_0:.*]], %[[VAL_1:.*]]) = (%[[CONSTANT_4]]) to (%[[CONSTANT_6]]) step (%[[CONSTANT_5]]) distribute(num_instances = 2) {
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:         ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:           "test.body"(%[[VAL_0]]) : (index) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.parallel_yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/






// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/






// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/






// ktdf.parallel construct test
// Tests: Presence of ktdf.parallel in the program with pipeline transformation happening
// only for non-parallel stages. The parallel region is preserved as-is.

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @basic_1d_balanced() attributes {grid = [2]}{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    ktdf.parallel (%arg0, %arg1) = (%c0) to (%c4) step (%c1) distribute(num_instances = 2) {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%arg0) : (index) -> ()
        } {applicable_units = ["SFU"]}
      }
      ktdf.parallel_yield
    }
    return
  }
}

