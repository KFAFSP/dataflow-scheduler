// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck %s
// Round-tripping test to ensure pipeline and stage operations are registered properly

// CHECK-LABEL:   func.func @pipeline_with_single_stage() {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage {
// CHECK-NEXT:         %[[VAL_0:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:         "test.op"(%[[VAL_0]]) : (!ktdf.token) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage {
// CHECK-NEXT:         %[[VAL_1:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:         "test.op"(%[[VAL_1]]) : (!ktdf.token) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @pipeline_with_multiple_stages() {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage {
// CHECK-NEXT:         %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage {
// CHECK-NEXT:         %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage {
// CHECK-NEXT:         %[[VAL_2:.*]] = arith.constant 3 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @nested_pipeline_operations() {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage {
// CHECK-NEXT:         %[[VAL_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>
// CHECK-NEXT:         "test.op"(%[[VAL_0]]#0) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>) -> ()
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           ktdf.stage {
// CHECK-NEXT:             %[[VAL_1:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:             "test.op"(%[[VAL_1]]) : (!ktdf.token) -> ()
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage {
// CHECK-NEXT:         %[[VAL_2:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:         "test.op"(%[[VAL_2]]) : (!ktdf.token) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @pipeline_with_single_stage() {
    ktdf.pipeline {
      ktdf.stage {
        %token = "test.op"() : () -> !ktdf.token
        "test.op"(%token) : (!ktdf.token) -> ()
      }
      ktdf.stage {
        %token = "test.op"() : () -> !ktdf.token
        "test.op"(%token) : (!ktdf.token) -> ()
      }
    }
    return
  }

  func.func @pipeline_with_multiple_stages() {
    // Pipeline with three stages
    ktdf.pipeline {
      ktdf.stage {
        %0 = arith.constant 1 : i32
      }
      ktdf.stage {
        %1 = arith.constant 2 : i32
      }
      ktdf.stage {
        %2 = arith.constant 3 : i32
      }
    }
    return
  }

  func.func @nested_pipeline_operations() {
    ktdf.pipeline {
      ktdf.stage {
        %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>
        "test.op"(%slot0) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>) -> ()
        ktdf.pipeline {
            ktdf.stage {
                %token = "test.op"() : () -> !ktdf.token
                "test.op"(%token) : (!ktdf.token) -> ()
            }
        }
      }
      ktdf.stage {
        %token = "test.op"() : () -> !ktdf.token
        "test.op"(%token) : (!ktdf.token) -> ()
      }
    }
    return
  }
}