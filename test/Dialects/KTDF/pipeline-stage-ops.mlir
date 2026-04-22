// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck --check-prefix=CHECK-IR %s
// Round-tripping test to ensure pipeline and stage operations are registered properly

// CHECK-IR-LABEL:   func.func @pipeline_with_no_dependencies() {
// CHECK-IR-NEXT:     ktdf.pipeline {
// CHECK-IR-NEXT:       ktdf.stage depends_in(none) depends_out(none) {
// CHECK-IR-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:     }
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }

// CHECK-IR-LABEL:   func.func @pipeline_with_single_stage() {
// CHECK-IR-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-IR-NEXT:     ktdf.pipeline {
// CHECK-IR-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]] : !ktdf.token) {
// CHECK-IR-NEXT:         %[[VAL_0:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-IR-NEXT:         "test.op"(%[[VAL_0]]) : (!ktdf.token) -> ()
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_0]] : !ktdf.token) depends_out(none) {
// CHECK-IR-NEXT:         %[[VAL_1:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-IR-NEXT:         "test.op"(%[[VAL_1]]) : (!ktdf.token) -> ()
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:     }
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }

// CHECK-IR-LABEL:   func.func @pipeline_with_multiple_stages() {
// CHECK-IR-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-IR-NEXT:     %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-IR-NEXT:     ktdf.pipeline {
// CHECK-IR-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]] : !ktdf.token) {
// CHECK-IR-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_0]] : !ktdf.token) depends_out(%[[CREATE_TOKEN_1]] : !ktdf.token) {
// CHECK-IR-NEXT:         %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_1]] : !ktdf.token) depends_out(none) {
// CHECK-IR-NEXT:         %[[CONSTANT_2:.*]] = arith.constant 3 : i32
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:     }
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }

// CHECK-IR-LABEL:   func.func @nested_pipeline_operations() {
// CHECK-IR-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-IR-NEXT:     %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-IR-NEXT:     ktdf.pipeline {
// CHECK-IR-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]] : !ktdf.token) {
// CHECK-IR-NEXT:         %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>
// CHECK-IR-NEXT:         "test.op"(%[[FIFO_0]]#0) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>) -> ()
// CHECK-IR-NEXT:         ktdf.pipeline {
// CHECK-IR-NEXT:           ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_1]] : !ktdf.token) {
// CHECK-IR-NEXT:             %[[VAL_0:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-IR-NEXT:             "test.op"(%[[VAL_0]]) : (!ktdf.token) -> ()
// CHECK-IR-NEXT:           }
// CHECK-IR-NEXT:         }
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_0]] : !ktdf.token) depends_out(none) {
// CHECK-IR-NEXT:         %[[VAL_1:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-IR-NEXT:         "test.op"(%[[VAL_1]]) : (!ktdf.token) -> ()
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:     }
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }


module {
  func.func @pipeline_with_no_dependencies() {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        %c = arith.constant 42 : i32
      }
    }
    return
  }

  func.func @pipeline_with_single_stage() {
    %token = ktdf.create_token : !ktdf.token
    ktdf.pipeline {
      ktdf.stage depends_out(%token : !ktdf.token) {
        %t = "test.op"() : () -> !ktdf.token
        "test.op"(%t) : (!ktdf.token) -> ()
      }
      ktdf.stage depends_in(%token : !ktdf.token) {
        %t = "test.op"() : () -> !ktdf.token
        "test.op"(%t) : (!ktdf.token) -> ()
      }
    }
    return
  }

  func.func @pipeline_with_multiple_stages() {
    %token_a = ktdf.create_token : !ktdf.token
    %token_b = ktdf.create_token : !ktdf.token
    
    ktdf.pipeline {
      ktdf.stage depends_out(%token_a : !ktdf.token) {
        %0 = arith.constant 1 : i32
      }
      ktdf.stage depends_in(%token_a : !ktdf.token) depends_out(%token_b : !ktdf.token) {
        %1 = arith.constant 2 : i32
      }
      ktdf.stage depends_in(%token_b : !ktdf.token) {
        %2 = arith.constant 3 : i32
      }
    }
    return
  }

  func.func @nested_pipeline_operations() {
    %token_outer = ktdf.create_token : !ktdf.token
    %token_inner = ktdf.create_token : !ktdf.token
    
    ktdf.pipeline {
      ktdf.stage depends_out(%token_outer : !ktdf.token){
        %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>
        "test.op"(%slot0) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>) -> ()
        ktdf.pipeline {
          ktdf.stage depends_out(%token_inner : !ktdf.token) {
            %token = "test.op"() : () -> !ktdf.token
            "test.op"(%token) : (!ktdf.token) -> ()
          }
        }
      }
      ktdf.stage depends_in(%token_outer : !ktdf.token) {
        %token = "test.op"() : () -> !ktdf.token
        "test.op"(%token) : (!ktdf.token) -> ()
      }
    }
    return
  }
}