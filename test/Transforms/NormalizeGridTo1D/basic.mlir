// RUN: dataflow-scheduler-opt -normalize-grid-to-1d -split-input-file %s | FileCheck %s

// 2-D grid: flattened to [24]; tile id becomes single-result; coords
// reconstructed row-major (coord0 = flat / 6, coord1 = flat % 6).
// CHECK-LABEL: func.func @grid_2d
// CHECK-SAME:    attributes {grid = [24{{.*}}]}
// CHECK:         %[[FLAT:.*]] = ktdp.get_compute_tile_id : index
// CHECK-DAG:     %[[C6A:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C6B:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[D0:.*]] = arith.divui %[[FLAT]], %[[C6A]] : index
// CHECK-DAG:     %[[R1:.*]] = arith.remui %[[FLAT]], %[[C6B]] : index
// CHECK:         arith.muli %[[D0]], %{{.*}}
// CHECK:         arith.muli %[[R1]], %{{.*}}
func.func @grid_2d() attributes {grid = [4 : index, 6 : index]} {
  %0:2 = ktdp.get_compute_tile_id : index, index
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index
  %1 = arith.muli %0#0, %c16 : index
  %2 = arith.muli %0#1, %c2 : index
  return
}

// -----

// 1-D grid: unchanged, no divui/remui introduced.
// CHECK-LABEL: func.func @grid_1d
// CHECK-SAME:    attributes {grid = [8{{.*}}]}
// CHECK:         %{{.*}} = ktdp.get_compute_tile_id : index
// CHECK-NOT:     arith.divui
// CHECK-NOT:     arith.remui
func.func @grid_1d() attributes {grid = [8 : index]} {
  %0 = ktdp.get_compute_tile_id : index
  return
}

// -----

// 3-D grid [2,3,4] -> flattened to [24]; row-major delinearization:
// coord0 = flat / 12, coord1 = (flat / 4) % 3, coord2 = flat % 4.
// CHECK-LABEL: func.func @grid_3d
// CHECK-SAME:    attributes {grid = [24{{.*}}]}
// CHECK:         %[[FLAT:.*]] = ktdp.get_compute_tile_id : index
// CHECK-DAG:     %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG:     %[[D0:.*]] = arith.divui %[[FLAT]], %[[C12]] : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[D1:.*]] = arith.divui %[[FLAT]], %[[C4]] : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[R1:.*]] = arith.remui %[[D1]], %[[C3]] : index
// CHECK-DAG:     %[[C4B:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[R2:.*]] = arith.remui %[[FLAT]], %[[C4B]] : index
func.func @grid_3d() attributes {grid = [2 : index, 3 : index, 4 : index]} {
  %0:3 = ktdp.get_compute_tile_id : index, index, index
  %c1 = arith.constant 1 : index
  %1 = arith.muli %0#0, %c1 : index
  %2 = arith.muli %0#1, %c1 : index
  %3 = arith.muli %0#2, %c1 : index
  return
}
