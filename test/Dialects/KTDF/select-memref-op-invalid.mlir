// RUN: not scheduler %s 2>&1 | FileCheck %s

// Negative: only 1 candidate.
// CHECK: 'ktdf.select_memref' op requires at least 2 candidate memrefs

func.func @select_memref_one_candidate() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    %sel = ktdf.select_memref %phase[%x0] : index, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}

// Negative: candidates with different types.
// CHECK: 'ktdf.select_memref' op candidate #1 type 'memref<32xf16, #ktdf.memory_space<LX>>' does not match

func.func @select_memref_mismatched_types() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<32xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<32xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<32xf16, #ktdf.memory_space<LX>>
  return
}

// Negative: phase not from buffer_phase.
// CHECK: 'ktdf.select_memref' op phase operand must be defined by a ktdf.buffer_phase op

func.func @select_memref_phase_from_constant() {
  %c0 = arith.constant 0 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %sel = ktdf.select_memref %c0[%x0, %x1] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}

// Negative: num_phases mismatches candidate count.
// CHECK: 'ktdf.select_memref' op phase op num_phases (2) must equal candidate count (3)

func.func @select_memref_num_phases_mismatch() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x2 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1, %x2] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x2 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}
