// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck %s

// Round trip test for ktdf.read_from_fifo

// CHECK-LABEL:   func.func @read_from_fifo_static() {
// CHECK-NEXT:     %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:     %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_0]] : <"lxlu-to-sfp", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:     %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>
// CHECK-NEXT:     %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[FIFO_1]] : <"sfp-to-lxsu", 8xf32> -> tensor<8xf32>
// CHECK-NEXT:     "test.op"(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]]) : (tensor<64xf16>, tensor<8xf32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  func.func @read_from_fifo_static() {
    // Allocate a FIFO slot with 64 f16 elements
    %slot0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
    
    // Read from the FIFO slot into a tensor
    %data0 = ktdf.read_from_fifo %slot0 : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16> -> tensor<64xf16>
    
    // Allocate another FIFO slot with 8 f32 elements
    %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>
    
    // Read from the second FIFO slot
    %data1 = ktdf.read_from_fifo %slot1 : !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32> -> tensor<8xf32>
    
    // Use the tensors
    "test.op"(%data0, %data1) : (tensor<64xf16>, tensor<8xf32>) -> ()
    
    return
  }
}