// RUN: dataflow-scheduler-opt %s \
// RUN:   --linalg-morph-ops="named-to-generic=true category-to-generic=true" \
// RUN:   --simplify-linalg | FileCheck %s

// Tests that generalized `linalg.fill` ops reduce to `arith.constant` whenever
// possible, and that invariant scalar loads and stores are hoisted out of
// `linalg.generic` ops.

#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @fill_f32
// CHECK-NOT:     linalg.generic
// CHECK:         %[[C:.+]] = arith.constant dense<1.000000e+00> : tensor<8xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[C]]
func.func @fill_f32() -> tensor<8xf32> {
  %c = arith.constant 1.0 : f32
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @fill_i32
// CHECK:         %[[C:.+]] = arith.constant dense<7> : tensor<4xi32>
// CHECK:         return %[[C]]
func.func @fill_i32() -> tensor<4xi32> {
  %c = arith.constant 7 : i32
  %e = tensor.empty() : tensor<4xi32>
  %0 = linalg.fill ins(%c : i32) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @fill_index
// CHECK:         %[[C:.+]] = arith.constant dense<3> : tensor<4xindex>
// CHECK:         return %[[C]]
func.func @fill_index() -> tensor<4xindex> {
  %c = arith.constant 3 : index
  %e = tensor.empty() : tensor<4xindex>
  %0 = linalg.fill ins(%c : index) outs(%e : tensor<4xindex>)
      -> tensor<4xindex>
  return %0 : tensor<4xindex>
}

// CHECK-LABEL: func.func @fill_rank0
// CHECK:         %[[C:.+]] = arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK:         return %[[C]]
func.func @fill_rank0() -> tensor<f32> {
  %c = arith.constant 2.0 : f32
  %e = tensor.empty() : tensor<f32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// The fill overwrites the entire init, so its contents do not matter.
// CHECK-LABEL: func.func @fill_init
// CHECK:         %[[C:.+]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK:         return %[[C]]
func.func @fill_init(%init: tensor<8xf32>) -> tensor<8xf32> {
  %c = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%c : f32) outs(%init : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// The fill value only becomes constant through folding.
// CHECK-LABEL: func.func @fill_folded
// CHECK:         %[[C:.+]] = arith.constant dense<3.000000e+00> : tensor<8xf32>
// CHECK:         return %[[C]]
func.func @fill_folded() -> tensor<8xf32> {
  %a = arith.constant 1.0 : f32
  %b = arith.constant 2.0 : f32
  %c = arith.addf %a, %b : f32
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Dynamic shapes can not become constants.
// CHECK-LABEL: func.func @fill_dynamic
// CHECK:         %[[CST:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[E:.+]] = tensor.empty
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%[[E]] : tensor<?xf32>) {
// CHECK:           linalg.yield %[[CST]]
// CHECK:         }
func.func @fill_dynamic(%n: index) -> tensor<?xf32> {
  %c = arith.constant 1.0 : f32
  %e = tensor.empty(%n) : tensor<?xf32>
  %0 = linalg.fill ins(%c : f32) outs(%e : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// Non-constant fill values can not become constants.
// CHECK-LABEL: func.func @fill_nonconstant
// CHECK-SAME:    %[[V:.+]]: f32
// CHECK:         %[[E:.+]] = tensor.empty
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%[[E]] : tensor<8xf32>) {
// CHECK:           linalg.yield %[[V]]
// CHECK:         }
func.func @fill_nonconstant(%v: f32) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.fill ins(%v : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// A constant access of a tensor input is hoisted as a `tensor.extract`.
// CHECK-LABEL: func.func @hoist_tensor
// CHECK-SAME:    %[[A:.+]]: tensor<8x16xf32>, %[[B:.+]]: tensor<16xf32>
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[C5:.+]] = arith.constant 5 : index
// CHECK:         %[[X:.+]] = tensor.extract %[[A]][%[[C3]], %[[C5]]]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[B]] : tensor<16xf32>)
// CHECK:         ^bb0(%[[Y:.+]]: f32, %{{.+}}: f32):
// CHECK:           arith.addf %[[X]], %[[Y]]
// CHECK:         }
func.func @hoist_tensor(%a: tensor<8x16xf32>, %b: tensor<16xf32>)
    -> tensor<16xf32> {
  %e = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (3, 5)>, #id, #id],
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
// CHECK-LABEL: func.func @hoist_rank0
// CHECK-SAME:    %[[A:.+]]: tensor<f32>, %[[B:.+]]: tensor<16xf32>
// CHECK:         %[[X:.+]] = tensor.extract %[[A]][]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[B]] : tensor<16xf32>)
// CHECK:         ^bb0(%[[Y:.+]]: f32, %{{.+}}: f32):
// CHECK:           arith.mulf %[[X]], %[[Y]]
// CHECK:         }
func.func @hoist_rank0(%a: tensor<f32>, %b: tensor<16xf32>) -> tensor<16xf32> {
  %e = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {indexing_maps = [#scalar, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<f32>, tensor<16xf32>) outs(%e : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.mulf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// A scalar input is forwarded directly.
// CHECK-LABEL: func.func @hoist_scalar
// CHECK-SAME:    %[[V:.+]]: f32, %[[B:.+]]: tensor<16xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[B]] : tensor<16xf32>)
// CHECK:         ^bb0(%[[Y:.+]]: f32, %{{.+}}: f32):
// CHECK:           arith.subf %[[Y]], %[[V]]
// CHECK:         }
func.func @hoist_scalar(%v: f32, %b: tensor<16xf32>) -> tensor<16xf32> {
  %e = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {indexing_maps = [#scalar, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%v, %b : f32, tensor<16xf32>) outs(%e : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.subf %y, %x : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// An invariant store is hoisted into a separate fill.
// CHECK-LABEL: func.func @hoist_store
// CHECK-SAME:    %[[A:.+]]: tensor<8xf32>, %[[V:.+]]: f32
// CHECK:         %[[FILL:.+]] = linalg.generic
// CHECK-SAME:      outs(%{{.+}} : tensor<8xf32>) {
// CHECK:           linalg.yield %[[V]]
// CHECK:         }
// CHECK:         %[[R:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[A]] : tensor<8xf32>) outs(%{{.+}} : tensor<8xf32>) {
// CHECK:           %[[N:.+]] = arith.negf
// CHECK:           linalg.yield %[[N]]
// CHECK:         }
// CHECK:         return %[[R]], %[[FILL]]
func.func @hoist_store(%a: tensor<8xf32>, %v: f32)
    -> (tensor<8xf32>, tensor<8xf32>) {
  %e = tensor.empty() : tensor<8xf32>
  %0:2 = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel"]}
         ins(%a : tensor<8xf32>) outs(%e, %e : tensor<8xf32>, tensor<8xf32>) {
  ^bb0(%x: f32, %o0: f32, %o1: f32):
    %n = arith.negf %x : f32
    linalg.yield %n, %v : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// An invariant store of a constant becomes a constant.
// CHECK-LABEL: func.func @hoist_store_constant
// CHECK:         %[[C:.+]] = arith.constant dense<1.000000e+00> : tensor<8xf32>
// CHECK:         %[[R:.+]] = linalg.generic
// CHECK:         return %[[R]], %[[C]]
func.func @hoist_store_constant(%a: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>) {
  %c = arith.constant 1.0 : f32
  %e = tensor.empty() : tensor<8xf32>
  %0:2 = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel"]}
         ins(%a : tensor<8xf32>) outs(%e, %e : tensor<8xf32>, tensor<8xf32>) {
  ^bb0(%x: f32, %o0: f32, %o1: f32):
    %n = arith.negf %x : f32
    linalg.yield %n, %c : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}
