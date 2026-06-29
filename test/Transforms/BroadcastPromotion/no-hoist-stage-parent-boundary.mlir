// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Nested pipeline inside an outer stage (post-stage-coarsening shape).
// The outer stage is the scope boundary; hoist cannot escape it.
//
// The inner transfer is invariant in %n1 only, so it hoists to the block
// immediately inside the outer stage (between scf.for %m1 and scf.for
// %n1). The sibling pipeline does NOT appear outside the outer stage.

// CHECK-LABEL:   func.func @nested_boundary
// CHECK-SAME:        %[[A:[^,]+]]: memref<?xf16, "DDR">, %[[M:[^,]+]]: index, %[[N:[^,]+]]: index) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[OUTER_T:.*]] = ktdf.private -> (!ktdf.token) {
// CHECK-NEXT:         %[[OUTER_TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[OUTER_TOK]] : !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[OUTER_T]]) {
// CHECK-NEXT:         scf.for %[[IM1:.*]] = %[[C0]] to %[[M]] step %[[C1]] {
// CHECK-NEXT:           %[[NEW_L1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:           ktdf.pipeline {
// CHECK-NEXT:             ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:               ktdf.data_transfer from %[[A]][%[[IM1]]] size [64] to %[[NEW_L1]][%[[C0]]] size [64] : memref<?xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.for %[[IN1:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[INNER_PRIV:.*]]:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
// CHECK-NEXT:                 %[[ORIG_L1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:                 %[[INNER_TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[ORIG_L1]], %[[INNER_TOK]] : memref<64xf16, "L1">, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[INNER_PRIV]]#1) {
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(%[[INNER_PRIV]]#1) depends_out(none) {
// CHECK-NEXT:                 "test.use"(%[[NEW_L1]]) : (memref<64xf16, "L1">) -> ()
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @nested_boundary(%A: memref<?xf16, "DDR">,
                             %M: index, %N: index) {
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
              }
              ktdf.stage depends_in(%inner_t) depends_out(none) {
                "test.use"(%inner_l1) : (memref<64xf16, "L1">) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
