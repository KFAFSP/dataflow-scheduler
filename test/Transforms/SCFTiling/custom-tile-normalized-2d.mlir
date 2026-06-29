// RUN: dataflow-scheduler-opt --tile-scf-for-loops="tile-sizes=5,3" %s | FileCheck %s

// 2D normalized perfectly-nested loop: takes the new ktdf path.
// Two outer tile loops, two inner point loops, with tiling.derive_size per : index
// dimension and tiling.linearize_index replacing each bare IV use.
//
// CHECK-LABEL: func.func @tile_2d
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C23:.*]] = arith.constant 23 : index
// CHECK-DAG:     %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[NI:.*]] = arith.ceildivui %[[C23]], %[[C5]] : index
// CHECK:         scf.for %[[TI:.*]] = %{{.*}} to %[[NI]] step %{{.*}} {
// CHECK:           %[[NJ:.*]] = arith.ceildivui %[[C12]], %[[C3]] : index
// CHECK:           scf.for %[[TJ:.*]] = %{{.*}} to %[[NJ]] step %{{.*}} {
// CHECK-NEXT:        %[[TSI:.*]] = ktdf.tiling.derive_size [%[[TI]] : %[[C5]]], total_size = %[[C23]] : index
// CHECK-NEXT:        %[[TSJ:.*]] = ktdf.tiling.derive_size [%[[TJ]] : %[[C3]]], total_size = %[[C12]] : index
// CHECK-NEXT:        scf.for %[[PI:.*]] = %{{.*}} to %[[TSI]] step %{{.*}} {
// CHECK-NEXT:          scf.for %[[PJ:.*]] = %{{.*}} to %[[TSJ]] step %{{.*}} {
// CHECK-NEXT:            %[[SUBJ:.*]] = ktdf.tiling.linearize_index [%[[TJ]] : %[[C3]]], [%[[PJ]] : %{{.*}}] : index
// CHECK-NEXT:            %[[SUBI:.*]] = ktdf.tiling.linearize_index [%[[TI]] : %[[C5]]], [%[[PI]] : %{{.*}}] : index
// CHECK-NEXT:            %{{.*}} = arith.addi %[[SUBI]], %[[SUBJ]] : index
// CHECK-NOT:     affine.apply
// CHECK-NOT:     affine.min
func.func @tile_2d() {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c23 = arith.constant 23 : index
  %c12 = arith.constant 12 : index
  scf.for %i = %c0 to %c23 step %c1 {
    scf.for %j = %c0 to %c12 step %c1 {
      %sum = arith.addi %i, %j : index
    }
  }
  return
}
