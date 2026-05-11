// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK-LABEL:   func.func @test_single_applicable_unit() {
// CHECK-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[SFP]>}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_multiple_applicable_units() {
// CHECK-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 100 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[SFP, LXLU, L3LU]>}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_all_unit_types() {
// CHECK-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     %[[CREATE_TOKEN_4:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[SFP]>}
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_0]] : !ktdf.token) depends_out(%[[CREATE_TOKEN_1]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[LXLU]>}
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_1]] : !ktdf.token) depends_out(%[[CREATE_TOKEN_2]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[LXSU]>}
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_2]] : !ktdf.token) depends_out(%[[CREATE_TOKEN_3]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[L3LU]>}
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_3]] : !ktdf.token) depends_out(%[[CREATE_TOKEN_4]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_4:.*]] = arith.constant 4 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[L3SU]>}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Round tripping test for applicable_units attribute

func.func @test_single_applicable_unit() {
  %token = ktdf.create_token : !ktdf.token
  ktdf.pipeline {
    ktdf.stage depends_out(%token : !ktdf.token) {
      %c = arith.constant 42 : i32
    } {applicable_units = #ktdf.applicable_units<[SFP]>}
  }
  return
}

func.func @test_multiple_applicable_units() {
  %token = ktdf.create_token : !ktdf.token
  ktdf.pipeline {
    ktdf.stage depends_out(%token : !ktdf.token) {
      %c = arith.constant 100 : i32
    } {applicable_units = #ktdf.applicable_units<[SFP, LXLU, L3LU]>}
  }
  return
}

func.func @test_all_unit_types() {
  %t0 = ktdf.create_token : !ktdf.token
  %t1 = ktdf.create_token : !ktdf.token
  %t2 = ktdf.create_token : !ktdf.token
  %t3 = ktdf.create_token : !ktdf.token
  %t4 = ktdf.create_token : !ktdf.token
  
  ktdf.pipeline {
    ktdf.stage depends_out(%t0 : !ktdf.token) {
      %c0 = arith.constant 0 : i32
    } {applicable_units = #ktdf.applicable_units<[SFP]>}
    ktdf.stage depends_in(%t0 : !ktdf.token) depends_out(%t1 : !ktdf.token) {
      %c1 = arith.constant 1 : i32
    } {applicable_units = #ktdf.applicable_units<[LXLU]>}
    ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
      %c2 = arith.constant 2 : i32
    } {applicable_units = #ktdf.applicable_units<[LXSU]>}
    ktdf.stage depends_in(%t2 : !ktdf.token) depends_out(%t3 : !ktdf.token) {
      %c3 = arith.constant 3 : i32
    } {applicable_units = #ktdf.applicable_units<[L3LU]>}
    ktdf.stage depends_in(%t3 : !ktdf.token) depends_out(%t4 : !ktdf.token) {
      %c4 = arith.constant 4 : i32
    } {applicable_units = #ktdf.applicable_units<[L3SU]>}
  }
  return
}