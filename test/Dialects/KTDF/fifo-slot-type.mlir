// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck %s




// CHECK-LABEL:   module {
// CHECK:          func.func @fifo_slot_static_sizes() {
// CHECK-NEXT:       %[[VAL_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>
// CHECK-NEXT:       %[[VAL_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>
// CHECK-NEXT:       %[[VAL_2:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 2xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 16xf16>
// CHECK-NEXT:       "test.op"(%[[VAL_0]]#0, %[[VAL_0]]#1, %[[VAL_1]]) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>) -> ()
// CHECK-NEXT:       return
// CHECK-NEXT:     }

// CHECK:          func.func @fifo_slot_dynamic_sizes(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index) {
// CHECK-NEXT:       %[[VAL_2:.*]] = ktdf.fifo.allocate(%[[VAL_0]]) -> !ktdf.fifo.slot<"lxlu-to-sfp", ?xf16>
// CHECK-NEXT:       %[[VAL_3:.*]]:2 = ktdf.fifo.allocate(%[[VAL_1]]) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?xf32>, !ktdf.fifo.slot<"sfp-to-lxsu", 4xf32>
// CHECK-NEXT:       "test.op"(%[[VAL_2]], %[[VAL_3]]#0) : (!ktdf.fifo.slot<"lxlu-to-sfp", ?xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", ?xf32>) -> ()
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }






module {
  func.func @fifo_slot_static_sizes() {
    // Allocate two FIFO slots with 1 and 4 f16 elements
    %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>
    
    // Allocate a single FIFO slot with 8 f32 elements
    %slot2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>
    
    // Allocate multiple slots with different types and sizes
    %slot3, %slot4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 2xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 16xf16>
    
    // Use the slots
    "test.op"(%slot0, %slot1, %slot2) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 4xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 8xf32>) -> ()
    
    return
  }

  func.func @fifo_slot_dynamic_sizes(%size0: index, %size1: index) {
    // Allocate a FIFO slot with some dynamic number of f16 elements
    %slot0 = ktdf.fifo.allocate(%size0) -> !ktdf.fifo.slot<"lxlu-to-sfp", ?xf16>
    
    // Mix static and dynamic sizes
    %slot1, %slot2 = ktdf.fifo.allocate(%size1) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?xf32>, !ktdf.fifo.slot<"sfp-to-lxsu", 4xf32>
    
    // Use the slots
    "test.op"(%slot0, %slot1) : (!ktdf.fifo.slot<"lxlu-to-sfp", ?xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", ?xf32>) -> ()
    
    return
  }
}
