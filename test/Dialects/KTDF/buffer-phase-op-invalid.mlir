// RUN: not scheduler %s 2>&1 | FileCheck %s

// Negative: num_phases < 2.
// CHECK: 'ktdf.buffer_phase' op num_phases must be >= 2, got 1

func.func @buffer_phase_num_phases_too_small() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 1 : i64} : index
  }
  return
}

// Negative: empty operand list.
// CHECK: 'ktdf.buffer_phase' op requires at least one induction variable operand

func.func @buffer_phase_no_ivs() {
  %phase = ktdf.buffer_phase() {num_phases = 2 : i64} : index
  return
}

// Negative: operand is not an scf.for IV.
// CHECK: 'ktdf.buffer_phase' op operand #0 is not an scf.for induction variable

func.func @buffer_phase_not_an_iv() {
  %c0 = arith.constant 0 : index
  %phase = ktdf.buffer_phase(%c0) {num_phases = 2 : i64} : index
  return
}
