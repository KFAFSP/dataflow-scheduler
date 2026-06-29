// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s
// CHECK-LABEL: func.func @index_math
func.func @index_math(%id: index) -> index {
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %m = arith.muli %id, %c64 : index
  %a = arith.addi %m, %c2 : index
  return %a : index
}
