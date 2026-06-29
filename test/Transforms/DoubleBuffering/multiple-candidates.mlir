// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// Two L1 buffers in the same pipeline. Expected: two alloc pairs, ONE
// buffer_phase shared across two select_memref ops, modulo(size: 2)
// once on the pipeline.

// CHECK-LABEL:   func.func @two_buffers(
// CHECK-SAME:        %[[A_SRC:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.spyre_memory_space<HBM>>, %[[B_SRC:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.spyre_memory_space<HBM>>, %[[DST:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.spyre_memory_space<HBM>>) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      %[[A0:.+]] = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      %[[A1:.+]] = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      %[[B0:.+]] = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      %[[B1:.+]] = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      scf.for %[[I:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-NEXT:        %[[PHASE:.+]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK-NEXT:        %[[A_SEL:.+]] = ktdf.select_memref %[[PHASE]][%[[A0]], %[[A1]]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:        %[[B_SEL:.+]] = ktdf.select_memref %[[PHASE]][%[[B0]], %[[B1]]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:        ktdf.pipeline modulo(size : 2) {
// CHECK-NEXT:          %[[TKS:.+]]:3 = ktdf.private -> (!ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            %[[T1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T3:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            ktdf.private_yield %[[T1]], %[[T2]], %[[T3]] : !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(%[[TKS]]#0) {
// CHECK-NEXT:            ktdf.data_transfer from %[[A_SRC]][%[[C0]]] size [64] to %[[A_SEL]][%[[C0]]] size [64] : memref<64xf16, #ktdp.spyre_memory_space<HBM>>, memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:            ktdf.data_transfer from %[[B_SRC]][%[[C0]]] size [64] to %[[B_SEL]][%[[C0]]] size [64] : memref<64xf16, #ktdp.spyre_memory_space<HBM>>, memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#0) depends_out(%[[TKS]]#1) {
// CHECK-NEXT:            ktdf.data_transfer from %[[A_SEL]][%[[C0]]] size [64] to %[[DST]][%[[C0]]] size [64] : memref<64xf16, #ktdp.spyre_memory_space<LX>>, memref<64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#1) depends_out(%[[TKS]]#2) {
// CHECK-NEXT:            ktdf.data_transfer from %[[B_SEL]][%[[C0]]] size [64] to %[[DST]][%[[C0]]] size [64] : memref<64xf16, #ktdp.spyre_memory_space<LX>>, memref<64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[B0]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      memref.dealloc %[[B1]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      memref.dealloc %[[A0]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      memref.dealloc %[[A1]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.spyre_memory_space<HBM> = "DDR", #ktdp.spyre_memory_space<LX> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @two_buffers(%a_src: memref<64xf16, #ktdp.spyre_memory_space<HBM>>,
                       %b_src: memref<64xf16, #ktdp.spyre_memory_space<HBM>>,
                       %dst: memref<64xf16, #ktdp.spyre_memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %l1a, %l1b, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, #ktdp.spyre_memory_space<LX>>,
          memref<64xf16, #ktdp.spyre_memory_space<LX>>,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
        %b = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %b, %tk1, %tk2, %tk3
          : memref<64xf16, #ktdp.spyre_memory_space<LX>>,
            memref<64xf16, #ktdp.spyre_memory_space<LX>>,
            !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        ktdf.data_transfer from %a_src[%c0] size [64] to %l1a[%c0] size [64]
          : memref<64xf16, #ktdp.spyre_memory_space<HBM>>,
            memref<64xf16, #ktdp.spyre_memory_space<LX>>
        ktdf.data_transfer from %b_src[%c0] size [64] to %l1b[%c0] size [64]
          : memref<64xf16, #ktdp.spyre_memory_space<HBM>>,
            memref<64xf16, #ktdp.spyre_memory_space<LX>>
      }
      ktdf.stage depends_in(%t1) depends_out(%t2) {
        ktdf.data_transfer from %l1a[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdp.spyre_memory_space<LX>>,
            memref<64xf16, #ktdp.spyre_memory_space<HBM>>
      }
      ktdf.stage depends_in(%t2) depends_out(%t3) {
        ktdf.data_transfer from %l1b[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdp.spyre_memory_space<LX>>,
            memref<64xf16, #ktdp.spyre_memory_space<HBM>>
      }
    }
  }
  return
}
