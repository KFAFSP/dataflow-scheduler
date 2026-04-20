// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck --check-prefix=CHECK-IR %s




// CHECK-IR-LABEL:   module {
// CHECK-IR:          func.func @fifo_slot_static_sizes() {
// CHECK-IR-NEXT:       %[[VAL_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1>, !ktdf.fifo.slot<"lxlu-to-sfp", 4>
// CHECK-IR-NEXT:       %[[VAL_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8>
// CHECK-IR-NEXT:       %[[VAL_2:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 2>, !ktdf.fifo.slot<"sfp-to-lxsu", 16>
// CHECK-IR-NEXT:       "test.op"(%[[VAL_0]]#0, %[[VAL_0]]#1, %[[VAL_1]]) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1>, !ktdf.fifo.slot<"lxlu-to-sfp", 4>, !ktdf.fifo.slot<"sfp-to-lxsu", 8>) -> ()
// CHECK-IR-NEXT:       return
// CHECK-IR-NEXT:     }

// CHECK-IR:          func.func @fifo_slot_dynamic_sizes(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index) {
// CHECK-IR-NEXT:       %[[VAL_2:.*]] = ktdf.fifo.allocate(%[[VAL_0]]) -> !ktdf.fifo.slot<"lxlu-to-sfp", ?>
// CHECK-IR-NEXT:       %[[VAL_3:.*]]:2 = ktdf.fifo.allocate(%[[VAL_1]]) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?>, !ktdf.fifo.slot<"sfp-to-lxsu", 4>
// CHECK-IR-NEXT:       "test.op"(%[[VAL_2]], %[[VAL_3]]#0) : (!ktdf.fifo.slot<"lxlu-to-sfp", ?>, !ktdf.fifo.slot<"sfp-to-lxsu", ?>) -> ()
// CHECK-IR-NEXT:       return
// CHECK-IR-NEXT:     }
// CHECK-IR-NEXT:   }






module {
  func.func @fifo_slot_static_sizes() {
    // Allocate two FIFO slots with static sizes: 1 and 4
    %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1>, !ktdf.fifo.slot<"lxlu-to-sfp", 4>
    
    // Allocate a single FIFO slot with static size 8
    %slot2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 8>
    
    // Allocate multiple slots with different types and sizes
    %slot3, %slot4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 2>, !ktdf.fifo.slot<"sfp-to-lxsu", 16>
    
    // Use the slots
    "test.op"(%slot0, %slot1, %slot2) : (!ktdf.fifo.slot<"lxlu-to-sfp", 1>, !ktdf.fifo.slot<"lxlu-to-sfp", 4>, !ktdf.fifo.slot<"sfp-to-lxsu", 8>) -> ()
    
    return
  }

  func.func @fifo_slot_dynamic_sizes(%size0: index, %size1: index) {
    // Allocate a FIFO slot with dynamic size
    %slot0 = ktdf.fifo.allocate(%size0) -> !ktdf.fifo.slot<"lxlu-to-sfp", ?>
    
    // Mix static and dynamic sizes
    %slot1, %slot2 = ktdf.fifo.allocate(%size1) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?>, !ktdf.fifo.slot<"sfp-to-lxsu", 4>
    
    // Use the slots
    "test.op"(%slot0, %slot1) : (!ktdf.fifo.slot<"lxlu-to-sfp", ?>, !ktdf.fifo.slot<"sfp-to-lxsu", ?>) -> ()
    
    return
  }
}
