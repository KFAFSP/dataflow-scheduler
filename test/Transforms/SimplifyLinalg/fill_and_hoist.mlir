// RUN: dataflow-scheduler-opt %s \
// RUN:   --linalg-morph-ops="named-to-generic=true category-to-generic=true" \
// RUN:   --simplify-linalg | FileCheck %s

// Tests that generalized `linalg.fill` ops reduce to `arith.constant` whenever
// possible, and that invariant scalar loads are hoisted out of
// `linalg.generic` ops.

// CHECK-LABEL:   func.func @fill_f32
// CHECK-NOT:       linalg.generic
// CHECK:           %[[C:.+]] = arith.constant dense<1.000000e+00> : tensor<8x16xf32>
// CHECK-NEXT:      return %[[C]]
func.func @fill_f32() -> tensor<8x16xf32> {
  %cst = arith.constant 1.0 : f32
  %e = tensor.empty() : tensor<8x16xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%e : tensor<8x16xf32>)
      -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL:   func.func @fill_i32
// CHECK:           %[[C:.+]] = arith.constant dense<7> : tensor<4xi32>
// CHECK-NEXT:      return %[[C]]
func.func @fill_i32() -> tensor<4xi32> {
  %c = arith.constant 7 : i32
  %e = tensor.empty() : tensor<4xi32>
  %0 = linalg.fill ins(%c : i32) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL:   func.func @fill_index
// CHECK:           %[[C:.+]] = arith.constant dense<3> : tensor<4xindex>
// CHECK-NEXT:      return %[[C]]
func.func @fill_index() -> tensor<4xindex> {
  %c = arith.constant 3 : index
  %e = tensor.empty() : tensor<4xindex>
  %0 = linalg.fill ins(%c : index) outs(%e : tensor<4xindex>)
      -> tensor<4xindex>
  return %0 : tensor<4xindex>
}

// CHECK-LABEL:   func.func @fill_rank0
// CHECK:           %[[C:.+]] = arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:      return %[[C]]
func.func @fill_rank0() -> tensor<f32> {
  %c = arith.constant 2.0 : f32
  %e = tensor.empty() : tensor<f32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// The fill overwrites the entire init, so its contents do not matter.
// CHECK-LABEL:   func.func @fill_init
// CHECK:           %[[C:.+]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK-NEXT:      return %[[C]]
func.func @fill_init(%init: tensor<8xf32>) -> tensor<8xf32> {
  %c = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%c : f32) outs(%init : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// The fill value only becomes constant through folding.
// CHECK-LABEL:   func.func @fill_folded
// CHECK:           %[[C:.+]] = arith.constant dense<3.000000e+00> : tensor<8xf32>
// CHECK-NEXT:      return %[[C]]
func.func @fill_folded() -> tensor<8xf32> {
  %a = arith.constant 1.0 : f32
  %b = arith.constant 2.0 : f32
  %c = arith.addf %a, %b : f32
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Dynamic shapes can not become constants, but the fill value is still
// hoisted.
// CHECK-LABEL:   func.func @fill_dynamic
// CHECK:           %[[CST:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[E:.+]] = tensor.empty
// CHECK:           linalg.generic
// CHECK-SAME:        outs(%[[E]] : tensor<?xf32>)
// CHECK-NEXT:      ^bb0(%{{.+}}: f32):
// CHECK-NEXT:        linalg.yield %[[CST]] : f32
func.func @fill_dynamic(%n: index) -> tensor<?xf32> {
  %c = arith.constant 1.0 : f32
  %e = tensor.empty(%n) : tensor<?xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// Non-constant fill values can not become constants, but are still hoisted.
// CHECK-LABEL:   func.func @fill_nonconstant
// CHECK-SAME:      %[[V:.+]]: f32
// CHECK:           %[[E:.+]] = tensor.empty
// CHECK:           linalg.generic
// CHECK-SAME:        outs(%[[E]] : tensor<8xf32>)
// CHECK-NEXT:      ^bb0(%{{.+}}: f32):
// CHECK-NEXT:        linalg.yield %[[V]] : f32
func.func @fill_nonconstant(%v: f32) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%v : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// A constant access of a tensor input is hoisted as a `tensor.extract`.
// CHECK-LABEL:   func.func @hoist_tensor
// CHECK-SAME:      %[[A:.+]]: tensor<8x16xf32>, %[[B:.+]]: tensor<16xf32>
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:       %[[C5:.+]] = arith.constant 5 : index
// CHECK:           %[[X:.+]] = tensor.extract %[[A]][%[[C3]], %[[C5]]]
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[B]] : tensor<16xf32>)
// CHECK-NEXT:      ^bb0(%[[Y:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:        arith.addf %[[X]], %[[Y]]
func.func @hoist_tensor(%a: tensor<8x16xf32>, %b: tensor<16xf32>)
    -> tensor<16xf32> {
  %e = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (3, 5)>,
                                        affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> (d0)>],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8x16xf32>, tensor<16xf32>)
       outs(%e : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// A rank-0 tensor input is hoisted as a `tensor.extract` without indices.
// CHECK-LABEL:   func.func @hoist_rank0
// CHECK-SAME:      %[[A:.+]]: tensor<f32>, %[[B:.+]]: tensor<16xf32>
// CHECK:           %[[X:.+]] = tensor.extract %[[A]][]
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[B]] : tensor<16xf32>)
// CHECK-NEXT:      ^bb0(%[[Y:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:        arith.mulf %[[X]], %[[Y]]
func.func @hoist_rank0(%a: tensor<f32>, %b: tensor<16xf32>) -> tensor<16xf32> {
  %e = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>,
                                        affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> (d0)>],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<f32>, tensor<16xf32>) outs(%e : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.mulf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// A scalar input is forwarded directly.
// CHECK-LABEL:   func.func @hoist_scalar
// CHECK-SAME:      %[[V:.+]]: f32, %[[B:.+]]: tensor<16xf32>
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[B]] : tensor<16xf32>)
// CHECK-NEXT:      ^bb0(%[[Y:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:        arith.subf %[[Y]], %[[V]]
func.func @hoist_scalar(%v: f32, %b: tensor<16xf32>) -> tensor<16xf32> {
  %e = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>,
                                        affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> (d0)>],
                       iterator_types = ["parallel"]}
       ins(%v, %b : f32, tensor<16xf32>) outs(%e : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.subf %y, %x : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
