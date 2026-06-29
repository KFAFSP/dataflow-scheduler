// RUN: dataflow-scheduler-opt --tile-scf-for-loops="tile-sizes=5" %s | dataflow-scheduler-opt --tile-scf-for-loops="tile-sizes=3" | dataflow-scheduler-opt | FileCheck %s

// Chained tiling: tile a single normalized loop with tile-size 5, then tile
// the resulting point loop again with tile-size 3. The second pass tiles a
// loop whose upper bound is a tiling.derive_size value (not a constant), so the : index
// pass goes through the same normalized path; the resulting innermost
// tiling.linearize_index carries the full ancestry [%i0:3],[%i1:5],[%i2:1].

// CHECK-LABEL: func.func @chained
// CHECK:         %[[C23:.*]] = arith.constant 23 : index
// CHECK:         %[[C5:.*]] = arith.constant 5 : index
// CHECK:         %[[N0:.*]] = arith.ceildivui %[[C23]], %[[C5]] : index
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[N1:.*]] = arith.ceildivui %[[N0]], %[[C3]] : index
// CHECK:         scf.for %[[I0:.*]] = %{{.*}} to %[[N1]] step %{{.*}} {
// CHECK-NEXT:      %[[TS0:.*]] = ktdf.tiling.derive_size [%[[I0]] : %[[C3]]], total_size = %[[N0]] : index
// CHECK-NEXT:      scf.for %[[I1:.*]] = %{{.*}} to %[[TS0]] step %{{.*}} {
// CHECK-NEXT:        %[[SUB1:.*]] = ktdf.tiling.linearize_index [%[[I0]] : %[[C3]]], [%[[I1]] : %{{.*}}] : index
// CHECK-NEXT:        %[[TS1:.*]] = ktdf.tiling.derive_size [%[[SUB1]] : %[[C5]]], total_size = %[[C23]] : index
// CHECK-NEXT:        scf.for %[[I2:.*]] = %{{.*}} to %[[TS1]] step %{{.*}} {
// CHECK-NEXT:          %[[SUB2:.*]] = ktdf.tiling.linearize_index [%[[I0]] : %[[C3]]], [%[[I1]] : %[[C5]]], [%[[I2]] : %{{.*}}] : index
// CHECK-NEXT:          %{{.*}} = arith.index_cast %[[SUB2]] : index to i32
// CHECK-NOT:     affine.apply
// CHECK-NOT:     affine.min
module {
func.func @chained() {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c23 = arith.constant 23 : index
  scf.for %i = %c0 to %c23 step %c1 {
    %use = arith.index_cast %i : index to i32
  }
  return
}
}
