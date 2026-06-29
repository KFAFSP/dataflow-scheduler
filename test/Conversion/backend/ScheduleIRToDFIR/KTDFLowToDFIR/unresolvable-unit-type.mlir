// RUN: not dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s 2>&1 | FileCheck %s

// An execute_on whose unit operand traces to neither a dataflow.get_unit nor
// a uniform.query_map chain (here, a func.func block argument) must produce
// a diagnostic and fail the pass.

// CHECK: ktdflowering-to-dfir: could not resolve component type for unit operand of ktdf_lowering.execute_on

ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @bad_input(%bad_unit: index) {
  ktdf_lowering.execute_on %bad_unit {
    %0 = arith.constant 0 : index
  }
  return
}
