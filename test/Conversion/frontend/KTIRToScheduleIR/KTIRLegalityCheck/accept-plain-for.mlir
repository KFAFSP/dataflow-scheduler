// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s
// CHECK-LABEL: func.func @plain_loop
func.func @plain_loop() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
  }
  return
}
