// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s
func.func @while_loop(%arg0: i32) {
  // expected-error @+1 {{V1 does not support scf.while loops}}
  %r = scf.while (%a = %arg0) : (i32) -> i32 {
    %c = arith.constant 0 : i32
    %cond = arith.cmpi slt, %a, %c : i32
    scf.condition(%cond) %a : i32
  } do {
  ^bb0(%b: i32):
    scf.yield %b : i32
  }
  return
}
