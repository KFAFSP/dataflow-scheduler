// RUN: scheduler %s | scheduler | FileCheck %s

// Round-trip: memref-to-fifo with non-identity source subscript.
// CHECK-LABEL: func.func @transfer_with_affine_subscript
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}}[%{{.*}} + 3, %{{.*}} * 2]
// CHECK-SAME: size [1, 64]
// CHECK-SAME: to %{{.*}} size [64]
func.func @transfer_with_affine_subscript() {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  %i = arith.constant 1 : index
  %j = arith.constant 2 : index
  ktdf.data_transfer
    from %A[%i + 3, %j * 2] size [1, 64]
    to %fifo size [64]
    : memref<128x128xf16, "DRAM">, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  return
}

// Round-trip: fifo-to-memref with constant subscript (broadcast pattern).
// CHECK-LABEL: func.func @transfer_with_constant_subscript
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}} size [64]
// CHECK-SAME: to %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, 0]
// CHECK-SAME: size [1, 1, 1, 1]
func.func @transfer_with_constant_subscript(%a: index, %b: index, %c: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>
  %B = memref.alloc() : memref<12x1x64x64xf16, "DRAM">
  ktdf.data_transfer
    from %fifo size [64]
    to %B[%a, %b, %c, 0] size [1, 1, 1, 1]
    : !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>, memref<12x1x64x64xf16, "DRAM">
  return
}

// Round-trip: identity-form subscripts (existing pattern, preserved).
// CHECK-LABEL: func.func @transfer_identity
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}}[%{{.*}}, %{{.*}}] size [1, 64]
// CHECK-SAME: to %{{.*}}[%{{.*}}] size [64]
func.func @transfer_identity(%i: index, %j: index, %c0: index) {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %lx = memref.alloc() : memref<64xf16, "LX">
  ktdf.data_transfer
    from %A[%i, %j] size [1, 64]
    to %lx[%c0] size [64]
    : memref<128x128xf16, "DRAM">, memref<64xf16, "LX">
  return
}
