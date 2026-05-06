// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   func.func @write_to_fifo_static() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant dense<1.000000e+00> : tensor<64xf16>
// CHECK-NEXT:     %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:     ktdf.write_to_fifo %[[CONSTANT_0]], %[[FIFO_0]] : tensor<64xf16>, <"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant dense<2.000000e+00> : tensor<8xf32>
// CHECK-NEXT:     %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>
// CHECK-NEXT:     ktdf.write_to_fifo %[[CONSTANT_1]], %[[FIFO_1]] : tensor<8xf32>, <"sfp-to-lxsu", 8xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Round-trip test for write_to_fifo



module {
  func.func @write_to_fifo_static() {
    // Create data tensor with 64 f16 elements
    %data0 = arith.constant dense<1.0> : tensor<64xf16>
    
    // Allocate a FIFO slot with 64 f16 elements
    %slot0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
    
    // Write to the FIFO slot
    ktdf.write_to_fifo %data0, %slot0 : tensor<64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
    
    // Create another data tensor with 8 f32 elements
    %data1 = arith.constant dense<2.0> : tensor<8xf32>
    
    // Allocate another FIFO slot with 8 f32 elements
    %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>
    
    // Write to the second FIFO slot
    ktdf.write_to_fifo %data1, %slot1 : tensor<8xf32>, !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>
    
    return
  }
}