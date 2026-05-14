// RUN: scheduler %s | scheduler | FileCheck %s

// CHECK-LABEL:   func.func @test_single_applicable_unit() {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]] = ktdf.private -> (!ktdf.token) {
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[CREATE_TOKEN_0]] : !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[SFP]>}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_multiple_applicable_units() {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]] = ktdf.private -> (!ktdf.token) {
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[CREATE_TOKEN_0]] : !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0]] : !ktdf.token) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 100 : i32
// CHECK-NEXT:       } {applicable_units = #ktdf.applicable_units<[SFP, LXLU, L3LU]>}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



module {


  // Round tripping test for applicable_units attribute

  func.func @test_single_applicable_unit() {
    ktdf.pipeline {
      %token = ktdf.private -> (!ktdf.token) {
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %t : !ktdf.token
      }
      ktdf.stage depends_out(%token : !ktdf.token) {
        %c = arith.constant 42 : i32
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
    return
  }

  func.func @test_multiple_applicable_units() {
    ktdf.pipeline {
      %token = ktdf.private -> (!ktdf.token) {
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %t : !ktdf.token
      }
      ktdf.stage depends_out(%token : !ktdf.token) {
        %c = arith.constant 100 : i32
      } {applicable_units = #ktdf.applicable_units<[SFP, LXLU, L3LU]>}
    }
    return
  }
}