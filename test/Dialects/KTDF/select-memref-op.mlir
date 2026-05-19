// RUN: scheduler %s | scheduler | FileCheck %s

// Round-trip: select between 2 LX memrefs.
// CHECK-LABEL:   func.func @select_memref_2_lx
// CHECK:           %[[X0:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           %[[X1:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           scf.for %[[I:.*]] = {{.*}} {
// CHECK:             %[[PHASE:.*]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK:             %[[SEL:.*]] = ktdf.select_memref %[[PHASE]][%[[X0]], %[[X1]]] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>

func.func @select_memref_2_lx() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}

// Round-trip: select among 4 candidates with num_phases=4.
// CHECK-LABEL:   func.func @select_memref_4_lx
// CHECK:           %[[SEL:.*]] = ktdf.select_memref %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>

func.func @select_memref_4_lx() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x2 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x3 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 4 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1, %x2, %x3] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x2 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x3 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}
