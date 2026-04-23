// RUN: scheduler %s | scheduler | FileCheck %s




// CHECK-LABEL:   func.func @test_data_transfer_memref_to_memref() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-NEXT:     %[[ALLOC_1:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:     ktdf.data_transfer from %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_1]], %[[CONSTANT_2]]] to %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_data_transfer_memref_to_fifo() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:     %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:     ktdf.data_transfer from %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_1]]] to %[[FIFO_0]] size{{\[}}%[[CONSTANT_1]]] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_data_transfer_fifo_to_memref() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-NEXT:     ktdf.data_transfer from %[[FIFO_0]] to %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_1]]] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @test_data_transfer_in_pipeline() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-NEXT:     %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-NEXT:     %[[ALLOC_2:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:         %[[ALLOC_3:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:         %[[ALLOC_4:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_1]], %[[CONSTANT_2]]] to %[[ALLOC_3]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_1]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_1]], %[[CONSTANT_2]]] to %[[ALLOC_4]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:         %[[ALLOC_5:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:         %[[ALLOC_6:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:         %[[ALLOC_7:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:         %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
// CHECK-NEXT:         %[[ALLOC_8:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-NEXT:         %[[ALLOC_9:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-NEXT:         %[[ALLOC_10:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_5]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] to %[[FIFO_0]]#0 size{{\[}}%[[CONSTANT_2]]] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_6]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] to %[[FIFO_0]]#1 size{{\[}}%[[CONSTANT_2]]] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:         ktdf.data_transfer from %[[FIFO_0]]#0 to %[[ALLOC_8]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
// CHECK-NEXT:         ktdf.data_transfer from %[[FIFO_0]]#1 to %[[ALLOC_9]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_10]]{{\[}}%[[CONSTANT_0]]] to %[[FIFO_1]] size{{\[}}%[[CONSTANT_2]]] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
// CHECK-NEXT:         ktdf.data_transfer from %[[FIFO_1]] size{{\[}}%[[CONSTANT_2]]] to %[[ALLOC_7]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:         %[[ALLOC_11:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_11]]{{\[}}%[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_2]]] to %[[ALLOC_2]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] size{{\[}}%[[CONSTANT_1]], %[[CONSTANT_2]]] : memref<64xf16, "LX">, memref<1x64xf16, "DRAM">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }










func.func @test_data_transfer_memref_to_memref() {
  %c0 = arith.constant 0 : index
  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  ktdf.data_transfer from %A[%c0, %c0] size [1, 64] to %lx_A[%c0] size [64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
  return
}


func.func @test_data_transfer_memref_to_fifo() {
  %c0 = arith.constant 0 : index
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  %fifo_slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
  ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
  return
}


func.func @test_data_transfer_fifo_to_memref() {
  %c0 = arith.constant 0 : index
  %fifo_slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
  %sfp_A = memref.alloc() : memref<64xf16, "SFP_REG">
  ktdf.data_transfer from %fifo_slot to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
  return
}


func.func @test_data_transfer_in_pipeline() {
  %c0 = arith.constant 0 : index
  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %B = memref.alloc() : memref<1x64xf16, "DRAM">
  %C = memref.alloc() : memref<1x64xf16, "DRAM">
  
  ktdf.pipeline {
    ktdf.stage {
      %lx_A = memref.alloc() : memref<64xf16, "LX">
      %lx_B = memref.alloc() : memref<64xf16, "LX">
      ktdf.data_transfer from %A[%c0, %c0] size [1, 64] to %lx_A[%c0] size [64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
      ktdf.data_transfer from %B[%c0, %c0] size [1, 64] to %lx_B[%c0] size [64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
    }
    
    ktdf.stage {
      %lx_A = memref.alloc() : memref<64xf16, "LX">
      %lx_B = memref.alloc() : memref<64xf16, "LX">
      %lx_C = memref.alloc() : memref<64xf16, "LX">
      %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
      %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
      %sfp_A = memref.alloc() : memref<64xf16, "SFP_REG">
      %sfp_B = memref.alloc() : memref<64xf16, "SFP_REG">
      %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">
      
      ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot_a size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
      ktdf.data_transfer from %lx_B[%c0] size [64] to %fifo_slot_b size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
      ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
      ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
      ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
      ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
    }
    
    ktdf.stage {
      %lx_C = memref.alloc() : memref<64xf16, "LX">      
      ktdf.data_transfer from %lx_C[%c0] size [64] to %C[%c0, %c0] size [1, 64] : memref<64xf16, "LX">, memref<1x64xf16, "DRAM">
    }
  }

  return
}
