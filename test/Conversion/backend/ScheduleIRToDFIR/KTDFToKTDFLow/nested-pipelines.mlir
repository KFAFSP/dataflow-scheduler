// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)"  -allow-unregistered-dialect %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK-LABEL:   func.func @nested_pipelines(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf16, "DDR">,
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: index) attributes {grid = [2]} {
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu", type = "l1lu"} : index
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
// CHECK-NEXT:     ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:       %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:         scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[ARG1]] step %[[CONSTANT_5]] {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[ARG2]] step %[[CONSTANT_5]] {
// CHECK-NEXT:             ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:               %[[ALLOC_0:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:               %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:                 ktdf.data_transfer from %[[ARG0]]{{\[}}%[[VAL_0]]] size [64] to %[[ALLOC_0]]{{\[}}%[[CONSTANT_4]]] size [64] : memref<?xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf_lowering.signal %[[QUERY_MAP_0]], %[[QUERY_MAP_1]]
// CHECK-NEXT:               ktdf_lowering.execute_on %[[QUERY_MAP_1]] {
// CHECK-NEXT:                 "test.use"(%[[ALLOC_0]]) : (memref<64xf16, "L1">) -> ()
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
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






module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @nested_pipelines(%A: memref<?xf16, "DDR">,
                              %M: index, %N: index) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    ktdf.pipeline {
      %outer_t = ktdf.private -> (!ktdf.token) {
        %k = ktdf.create_token : !ktdf.token
        ktdf.private_yield %k : !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%outer_t) {
        scf.for %m1 = %c0 to %M step %c1 {
          scf.for %n1 = %c0 to %N step %c1 {
            ktdf.pipeline {
              %inner_l1, %inner_t = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
                %a = memref.alloc() : memref<64xf16, "L1">
                %k = ktdf.create_token : !ktdf.token
                ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%inner_t) {
                ktdf.data_transfer from %A[%m1] size [64] to %inner_l1[%c0] size [64]
                  : memref<?xf16, "DDR">, memref<64xf16, "L1">
              } {applicable_units = ["MNILU"]}
              ktdf.stage depends_in(%inner_t) depends_out(none) {
                "test.use"(%inner_l1) : (memref<64xf16, "L1">) -> ()
              } {applicable_units = ["L1LU"]}
            }
          }
        }
      } {applicable_units = ["MNILU", "L1LU"]}
    }
    return
  }
}

