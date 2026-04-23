// RUN: scheduler %s | scheduler | FileCheck %s

// Test data transfer with constant sizes (new syntax matching documentation)
// CHECK-LABEL: func.func @test_constant_sizes
func.func @test_constant_sizes() {
  %c0 = arith.constant 0 : index
  
  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  
  // Using constant sizes directly in the IR (matches documentation examples)
  // CHECK: ktdf.data_transfer from {{.*}} size [1, 64] to {{.*}} size [64]
  ktdf.data_transfer from %A[%c0, %c0] size [1, 64] to %lx_A[%c0] size [64] 
    : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
  
  return
}

// Test data transfer with SSA operand sizes (existing syntax - still supported)
// CHECK-LABEL: func.func @test_dynamic_sizes
func.func @test_dynamic_sizes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  
  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  
  // Using SSA values for sizes (existing behavior preserved)
  // CHECK: ktdf.data_transfer from {{.*}} size [{{.*}}, {{.*}}] to {{.*}} size [{{.*}}]
  ktdf.data_transfer from %A[%c0, %c0] size [%c1, %c64] to %lx_A[%c0] size [%c64]
    : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
  
  return
}

// Test data transfer with mixed constant and dynamic sizes
// CHECK-LABEL: func.func @test_mixed_sizes
func.func @test_mixed_sizes(%dynamic_size: index) {
  %c0 = arith.constant 0 : index
  
  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  
  // Mix of constant (1) and dynamic (%dynamic_size) - best of both worlds
  // CHECK: ktdf.data_transfer from {{.*}} size [1, {{.*}}] to {{.*}} size [{{.*}}]
  ktdf.data_transfer from %A[%c0, %c0] size [1, %dynamic_size] to %lx_A[%c0] size [%dynamic_size]
    : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
  
  return
}

// Test FIFO transfers with constant sizes
// CHECK-LABEL: func.func @test_fifo_constant_sizes
func.func @test_fifo_constant_sizes() {
  %c0 = arith.constant 0 : index
  
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  %fifo_slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
  
  // FIFO transfer with constant size
  // CHECK: ktdf.data_transfer from {{.*}} size [64] to {{.*}} size [64]
  ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot size [64]
    : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
  
  return
}