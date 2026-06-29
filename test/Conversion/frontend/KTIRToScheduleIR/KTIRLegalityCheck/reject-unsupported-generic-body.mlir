// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s
#map = affine_map<(d0) -> (d0)>
func.func @bad_body(%a: tensor<4xf16>, %o: tensor<4xf16>) -> tensor<4xf16> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<4xf16>) outs(%o : tensor<4xf16>) {
  ^bb0(%in: f16, %out: f16):
    // expected-error @+1 {{V1 only supports add/mul/sub compute ops; found unsupported compute op}}
    %s = math.sqrt %in : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %r : tensor<4xf16>
}
