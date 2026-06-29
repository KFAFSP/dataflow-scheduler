// RUN: dataflow-scheduler-opt --tile-scf-for-loops="tile-sizes=5" %s | FileCheck %s

// Non-normalized loop (lb=1): falls back to mlir::tilePerfectlyNested.
// CHECK-LABEL: func.func @non_normalized
// CHECK-NOT: ktdf.tiling.derive_size
func.func @non_normalized(%N: index) {
  %c1 = arith.constant 1 : index
  scf.for %i = %c1 to %N step %c1 {
    %use = arith.addi %i, %c1 : index
  }
  return
}

// Normalized 1D loop (lb=0, step=1) with bare IV use: takes the new ktdf path.
// CHECK-LABEL: func.func @one_d_normalized
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C23:.*]] = arith.constant 23 : index
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK:         %[[NTILES:.*]] = arith.ceildivui %[[C23]], %[[C5]] : index
// CHECK:         scf.for %[[TI:.*]] = %{{.*}} to %[[NTILES]] step %{{.*}} {
// CHECK-NEXT:      %[[TS:.*]] = ktdf.tiling.derive_size [%[[TI]] : %[[C5]]], total_size = %[[C23]] : index
// CHECK-NEXT:      scf.for %[[PI:.*]] = %{{.*}} to %[[TS]] step %{{.*}} {
// CHECK-NEXT:        %[[SUB:.*]] = ktdf.tiling.linearize_index [%[[TI]] : %[[C5]]], [%[[PI]] : %{{.*}}] : index
// CHECK-NEXT:        %{{.*}} = arith.index_cast %[[SUB]] : index to i32
// CHECK-NOT:     affine.apply
// CHECK-NOT:     affine.min
func.func @one_d_normalized() {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c23 = arith.constant 23 : index
  scf.for %i = %c0 to %c23 step %c1 {
    %use = arith.index_cast %i : index to i32
  }
  return
}

// Normalized 1D loop with no IV use: still tiles, no tiling.linearize_index needed.
// CHECK-LABEL: func.func @one_d_no_iv_use
// CHECK-DAG:     %[[NC0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[NC1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[NC23:.*]] = arith.constant 23 : index
// CHECK-DAG:     %[[NC5:.*]] = arith.constant 5 : index
// CHECK:         %[[NN:.*]] = arith.ceildivui %[[NC23]], %[[NC5]] : index
// CHECK:         scf.for %[[NTI:.*]] = %{{.*}} to %[[NN]] step %{{.*}} {
// CHECK-NEXT:      %[[NTS:.*]] = ktdf.tiling.derive_size [%[[NTI]] : %[[NC5]]], total_size = %[[NC23]] : index
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %[[NTS]] step %{{.*}} {
// CHECK-NOT:     ktdf.tiling.linearize_index
// CHECK-NOT:     affine.apply
// CHECK-NOT:     affine.min
func.func @one_d_no_iv_use() {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c23 = arith.constant 23 : index
  scf.for %i = %c0 to %c23 step %c1 {
    %x = arith.constant 7 : index
  }
  return
}
