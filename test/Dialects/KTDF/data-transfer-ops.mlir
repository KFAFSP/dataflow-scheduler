// RUN: scheduler %s | scheduler | FileCheck --check-prefix=CHECK-IR %s




// CHECK-IR-LABEL:   func.func @test_data_transfer_memref_to_memref() {
// CHECK-IR-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-IR-NEXT:     %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-IR-NEXT:     %[[VAL_2:.*]] = arith.constant 64 : index
// CHECK-IR-NEXT:     %[[VAL_3:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-IR-NEXT:     %[[VAL_4:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:     ktdf.data_transfer from %[[VAL_3]]{{\[}}%[[VAL_0]], %[[VAL_0]]] size{{\[}}%[[VAL_1]], %[[VAL_2]]] to %[[VAL_4]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }

// CHECK-IR-LABEL:   func.func @test_data_transfer_memref_to_fifo() {
// CHECK-IR-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-IR-NEXT:     %[[VAL_1:.*]] = arith.constant 64 : index
// CHECK-IR-NEXT:     %[[VAL_2:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:     %[[VAL_3:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-IR-NEXT:     ktdf.data_transfer from %[[VAL_2]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_1]]] to %[[VAL_3]] size{{\[}}%[[VAL_1]]] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }

// CHECK-IR-LABEL:   func.func @test_data_transfer_fifo_to_memref() {
// CHECK-IR-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-IR-NEXT:     %[[VAL_1:.*]] = arith.constant 64 : index
// CHECK-IR-NEXT:     %[[VAL_2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-IR-NEXT:     %[[VAL_3:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-IR-NEXT:     ktdf.data_transfer from %[[VAL_2]] to %[[VAL_3]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_1]]] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }

// CHECK-IR-LABEL:   func.func @test_data_transfer_in_pipeline() {
// CHECK-IR-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-IR-NEXT:     %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-IR-NEXT:     %[[VAL_2:.*]] = arith.constant 64 : index
// CHECK-IR-NEXT:     %[[VAL_3:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-IR-NEXT:     %[[VAL_4:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-IR-NEXT:     %[[VAL_5:.*]] = memref.alloc() : memref<1x64xf16, "DRAM">
// CHECK-IR-NEXT:     ktdf.pipeline {
// CHECK-IR-NEXT:       ktdf.stage {
// CHECK-IR-NEXT:         %[[VAL_6:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:         %[[VAL_7:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_3]]{{\[}}%[[VAL_0]], %[[VAL_0]]] size{{\[}}%[[VAL_1]], %[[VAL_2]]] to %[[VAL_6]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_4]]{{\[}}%[[VAL_0]], %[[VAL_0]]] size{{\[}}%[[VAL_1]], %[[VAL_2]]] to %[[VAL_7]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:       ktdf.stage {
// CHECK-IR-NEXT:         %[[VAL_8:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:         %[[VAL_9:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:         %[[VAL_10:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:         %[[VAL_11:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-IR-NEXT:         %[[VAL_12:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
// CHECK-IR-NEXT:         %[[VAL_13:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-IR-NEXT:         %[[VAL_14:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-IR-NEXT:         %[[VAL_15:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_8]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] to %[[VAL_11]]#0 size{{\[}}%[[VAL_2]]] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_9]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] to %[[VAL_11]]#1 size{{\[}}%[[VAL_2]]] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_11]]#0 to %[[VAL_13]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_11]]#1 to %[[VAL_14]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_15]]{{\[}}%[[VAL_0]]] to %[[VAL_12]] size{{\[}}%[[VAL_2]]] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_12]] size{{\[}}%[[VAL_2]]] to %[[VAL_10]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:       ktdf.stage {
// CHECK-IR-NEXT:         %[[VAL_16:.*]] = memref.alloc() : memref<64xf16, "LX">
// CHECK-IR-NEXT:         ktdf.data_transfer from %[[VAL_16]]{{\[}}%[[VAL_0]]] size{{\[}}%[[VAL_2]]] to %[[VAL_5]]{{\[}}%[[VAL_0]], %[[VAL_0]]] size{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<64xf16, "LX">, memref<1x64xf16, "DRAM">
// CHECK-IR-NEXT:       }
// CHECK-IR-NEXT:     }
// CHECK-IR-NEXT:     return
// CHECK-IR-NEXT:   }


func.func @test_data_transfer_memref_to_memref() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  
  ktdf.data_transfer from %A[%c0, %c0] size [%c1, %c64] to %lx_A[%c0] size [%c64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">

  return
}


func.func @test_data_transfer_memref_to_fifo() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index

  %lx_A = memref.alloc() : memref<64xf16, "LX">
  %fifo_slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
  
  ktdf.data_transfer from %lx_A[%c0] size [%c64] to %fifo_slot size [%c64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>

  return
}


func.func @test_data_transfer_fifo_to_memref() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index

  %fifo_slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
  %sfp_A = memref.alloc() : memref<64xf16, "SFP_REG">

  ktdf.data_transfer from %fifo_slot to %sfp_A[%c0] size [%c64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">

  return
}


func.func @test_data_transfer_in_pipeline() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %B = memref.alloc() : memref<1x64xf16, "DRAM">
  %C = memref.alloc() : memref<1x64xf16, "DRAM">
  
  ktdf.pipeline {
    ktdf.stage {
      %lx_A = memref.alloc() : memref<64xf16, "LX">
      %lx_B = memref.alloc() : memref<64xf16, "LX">
      ktdf.data_transfer from %A[%c0, %c0] size [%c1, %c64] to %lx_A[%c0] size [%c64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
      ktdf.data_transfer from %B[%c0, %c0] size [%c1, %c64] to %lx_B[%c0] size [%c64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
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
      
      ktdf.data_transfer from %lx_A[%c0] size [%c64] to %fifo_slot_a size [%c64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
      ktdf.data_transfer from %lx_B[%c0] size [%c64] to %fifo_slot_b size [%c64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
      ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [%c64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
      ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [%c64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
      ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [%c64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
      ktdf.data_transfer from %fifo_slot_c size [%c64] to %lx_C[%c0] size [%c64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
    }
    
    ktdf.stage {
      %lx_C = memref.alloc() : memref<64xf16, "LX">      
      ktdf.data_transfer from %lx_C[%c0] size [%c64] to %C[%c0, %c0] size [%c1, %c64] : memref<64xf16, "LX">, memref<1x64xf16, "DRAM">
    }
  }

  return
}
