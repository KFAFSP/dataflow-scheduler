// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK-LABEL:   func.func @basic_1d_balanced() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-SFU-CL0", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-SFU-CL0", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-SFU-CL1", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-SFU-CL1", type = "SFU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]], %[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:         "test.body"() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }






// Same component type with multiple unit Values across the parallel construct.
// The corresponding program_unit's units variadic must list all of them in
// first-encountered order.


module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @basic_1d_balanced() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-SFU-CL0", type = "SFU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-SFU-CL0", type = "SFU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-SFU-CL1", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-SFU-CL1", type = "SFU"} : index
    %4 = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %5 = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %6 = uniform.query_map(map:%5, key:%4) : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %7 = uniform.def_immutable_mapping([%c0_0 -> %2], [%c1_1 -> %3]):index
    %8 = uniform.query_map(map:%7, key:%4) : index
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    ktdf.parallel (%arg0, %arg1) = (%c0_2) to (%c4) step (%c1_3) distribute(num_instances = 2) {
      ktdf_lowering.execute_on %6, %8 {
        ktdf_lowering.execute_on %6, %8 {
          "test.body"() : () -> ()
        }
      }
      ktdf.parallel_yield
    }
    return
  }
}

