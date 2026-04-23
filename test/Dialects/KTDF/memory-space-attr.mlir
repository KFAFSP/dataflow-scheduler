// RUN: scheduler %s | scheduler | FileCheck %s

// Round trip test for memory space attribute

// CHECK-LABEL: func.func @test_memory_space_attr
// CHECK: memref<4096xi32, #ktdf.memory_space<"HBM">>

func.func @test_memory_space_attr() {
  %1 = memref.alloc() : memref<4096xi32, #ktdf.memory_space<"HBM">>
  return
}