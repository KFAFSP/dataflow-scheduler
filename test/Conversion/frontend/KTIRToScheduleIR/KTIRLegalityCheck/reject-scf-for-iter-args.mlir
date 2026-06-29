// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s
func.func @loop_carried(%arg0: tensor<4xf16>) -> tensor<4xf16> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{V1 does not support scf.for with loop-carried arguments (iter_args)}}
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> tensor<4xf16> {
    scf.yield %acc : tensor<4xf16>
  }
  return %r : tensor<4xf16>
}
