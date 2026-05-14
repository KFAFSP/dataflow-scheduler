// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Two hoistable transfers in one stage with different deepest legal
// targets. The fixed-point loop picks the candidate with the deepest
// target each iteration (rather than first-in-source-order):
//   - First iteration: B (no IV deps) hoists past both %m and %n,
//     landing at the function-body scope above scf.for %m.
//   - Second iteration: A (depends on %m) hoists past %n only,
//     landing inside scf.for %m, before scf.for %n.

// CHECK-LABEL:   func.func @fixed_point
// CHECK-SAME:        %[[A:[^,]+]]: memref<?xf16, #ktdf.memory_space<HBM>>, %[[B:[^,]+]]: memref<?xf16, #ktdf.memory_space<HBM>>, %[[M:[^,]+]]: index, %[[N:[^,]+]]: index) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[NEW_LX_B:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:         ktdf.data_transfer from %[[B]][%[[C0]]] size [64] to %[[NEW_LX_B]][%[[C0]]] size [64] : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.for %[[IM:.*]] = %[[C0]] to %[[M]] step %[[C1]] {
// CHECK-NEXT:       %[[NEW_LX_A:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:           ktdf.data_transfer from %[[A]][%[[IM]]] size [64] to %[[NEW_LX_A]][%[[C0]]] size [64] : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.for %[[IN:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIV:.*]]:3 = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
// CHECK-NEXT:             %[[ORIG_LX_A:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:             %[[ORIG_LX_B:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:             %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ORIG_LX_A]], %[[ORIG_LX_B]], %[[TOK]] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIV]]#2 : !ktdf.token) {
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIV]]#2 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:             "test.use"(%[[NEW_LX_A]]) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
// CHECK-NEXT:             "test.use"(%[[NEW_LX_B]]) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @fixed_point(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                         %B: memref<?xf16, #ktdf.memory_space<HBM>>,
                         %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %lx_A, %lx_B, %t = ktdf.private -> (
              memref<64xf16, #ktdf.memory_space<LX>>,
              memref<64xf16, #ktdf.memory_space<LX>>,
              !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %b = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %b, %k
              : memref<64xf16, #ktdf.memory_space<LX>>,
                memref<64xf16, #ktdf.memory_space<LX>>,
                !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t : !ktdf.token) {
            // A: depends on %m only -> hoists past %n.
            ktdf.data_transfer from %A[%m] size [64] to %lx_A[%c0] size [64]
              : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
            // B: no IV deps -> hoists past both %m and %n.
            ktdf.data_transfer from %B[%c0] size [64] to %lx_B[%c0] size [64]
              : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
          }
          ktdf.stage depends_in(%t : !ktdf.token) depends_out(none) {
            "test.use"(%lx_A) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
            "test.use"(%lx_B) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
          }
        }
      }
    }
    return
  }
}
