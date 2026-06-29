// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ok_compute
func.func @ok_compute(%a: tensor<4xf16>, %b: tensor<4xf16>, %o: tensor<4xf16>) -> tensor<4xf16> {
  %r0 = linalg.add ins(%a, %b : tensor<4xf16>, tensor<4xf16>) outs(%o : tensor<4xf16>) -> tensor<4xf16>
  %r1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
        ins(%r0, %b : tensor<4xf16>, tensor<4xf16>) outs(%o : tensor<4xf16>) {
  ^bb0(%x: f16, %y: f16, %out: f16):
    %m = arith.mulf %x, %y : f16
    %s = arith.subf %m, %x : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %r1 : tensor<4xf16>
}
