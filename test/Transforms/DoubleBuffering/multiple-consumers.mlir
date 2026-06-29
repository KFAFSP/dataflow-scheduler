// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// One producer stage, two consumer stages reading the same buffer.

// CHECK-LABEL:   func.func @two_consumers(
// CHECK-SAME:        %[[SRC:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.spyre_memory_space<HBM>>, %[[DST1:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.spyre_memory_space<HBM>>, %[[DST2:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.spyre_memory_space<HBM>>) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      %[[X0:.+]] = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      %[[X1:.+]] = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      scf.for %[[I:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-NEXT:        %[[PHASE:.+]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK-NEXT:        %[[SEL:.+]] = ktdf.select_memref %[[PHASE]][%[[X0]], %[[X1]]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:        ktdf.pipeline modulo(size : 2) {
// CHECK-NEXT:          %[[TKS:.+]]:3 = ktdf.private -> (!ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            %[[T1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T3:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            ktdf.private_yield %[[T1]], %[[T2]], %[[T3]] : !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(%[[TKS]]#0) {
// CHECK-NEXT:            ktdf.data_transfer from %[[SRC]][%[[C0]]] size [64] to %[[SEL]][%[[C0]]] size [64] : memref<64xf16, #ktdp.spyre_memory_space<HBM>>, memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#0) depends_out(%[[TKS]]#1) {
// CHECK-NEXT:            ktdf.data_transfer from %[[SEL]][%[[C0]]] size [64] to %[[DST1]][%[[C0]]] size [64] : memref<64xf16, #ktdp.spyre_memory_space<LX>>, memref<64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#0) depends_out(%[[TKS]]#2) {
// CHECK-NEXT:            ktdf.data_transfer from %[[SEL]][%[[C0]]] size [64] to %[[DST2]][%[[C0]]] size [64] : memref<64xf16, #ktdp.spyre_memory_space<LX>>, memref<64xf16, #ktdp.spyre_memory_space<HBM>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[X0]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      memref.dealloc %[[X1]] : memref<64xf16, #ktdp.spyre_memory_space<LX>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.spyre_memory_space<HBM> = "DDR", #ktdp.spyre_memory_space<LX> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @two_consumers(%src: memref<64xf16, #ktdp.spyre_memory_space<HBM>>,
                         %dst1: memref<64xf16, #ktdp.spyre_memory_space<HBM>>,
                         %dst2: memref<64xf16, #ktdp.spyre_memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %l1, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, #ktdp.spyre_memory_space<LX>>,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdp.spyre_memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2, %tk3
          : memref<64xf16, #ktdp.spyre_memory_space<LX>>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        ktdf.data_transfer from %src[%c0] size [64] to %l1[%c0] size [64]
          : memref<64xf16, #ktdp.spyre_memory_space<HBM>>,
            memref<64xf16, #ktdp.spyre_memory_space<LX>>
      }
      ktdf.stage depends_in(%t1) depends_out(%t2) {
        ktdf.data_transfer from %l1[%c0] size [64] to %dst1[%c0] size [64]
          : memref<64xf16, #ktdp.spyre_memory_space<LX>>,
            memref<64xf16, #ktdp.spyre_memory_space<HBM>>
      }
      ktdf.stage depends_in(%t1) depends_out(%t3) {
        ktdf.data_transfer from %l1[%c0] size [64] to %dst2[%c0] size [64]
          : memref<64xf16, #ktdp.spyre_memory_space<LX>>,
            memref<64xf16, #ktdp.spyre_memory_space<HBM>>
      }
    }
  }
  return
}
