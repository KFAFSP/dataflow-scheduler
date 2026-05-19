// RUN: scheduler %s | scheduler | FileCheck %s

// Round-trip: 1 IV, num_phases = 2.
// CHECK-LABEL:   func.func @buffer_phase_1d
// CHECK:           %[[PHASE:.*]] = ktdf.buffer_phase(%[[IV:.*]]) {num_phases = 2 : i64} : index

func.func @buffer_phase_1d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
  }
  return
}

// Round-trip: 2 IVs, num_phases = 2.
// CHECK-LABEL:   func.func @buffer_phase_2d
// CHECK:           %[[PHASE:.*]] = ktdf.buffer_phase(%[[M:.*]], %[[N:.*]]) {num_phases = 2 : i64} : index

func.func @buffer_phase_2d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %m = %c0 to %c8 step %c1 {
    scf.for %n = %c0 to %c8 step %c1 {
      %phase = ktdf.buffer_phase(%m, %n) {num_phases = 2 : i64} : index
    }
  }
  return
}

// Round-trip: num_phases = 4 still parses.
// CHECK-LABEL:   func.func @buffer_phase_n4
// CHECK:           %{{.*}} = ktdf.buffer_phase(%{{.*}}) {num_phases = 4 : i64} : index

func.func @buffer_phase_n4() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 4 : i64} : index
  }
  return
}
