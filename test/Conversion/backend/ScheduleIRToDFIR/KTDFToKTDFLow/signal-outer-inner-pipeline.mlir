// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)" -allow-unregistered-dialect %s | FileCheck %s

// Regression test: signal between outer stage A (MNILU) and outer stage B
// (which wraps an inner pipeline L1LU->SFU->L1SU) must include only {MNILU, L1LU},
// not all inner units. Similarly, signal between B and outer stage C (MNISU)
// must include only {L1SU, MNISU}.

// CHECK-LABEL: func.func @signal_narrowing(
// Outer stage A (MNILU) unit mapping.
// CHECK:      %[[MNILU:.*]] = uniform.query_map
// Outer stage B entry unit (L1LU) mapping.
// CHECK:      %[[L1LU:.*]] = uniform.query_map
// Inner SFU unit mapping (present, but must NOT appear in signals).
// CHECK:      uniform.query_map
// Outer stage B exit unit (L1SU) mapping.
// CHECK:      %[[L1SU:.*]] = uniform.query_map
// Outer stage C (MNISU) unit mapping.
// CHECK:      %[[MNISU:.*]] = uniform.query_map
// Signal A→B: must involve exactly MNILU and L1LU (not SFU, not L1SU).
// CHECK:      ktdf_lowering.signal %[[MNILU]], %[[L1LU]]
// Signal B→C: must involve exactly L1SU and MNISU (not MNILU, not L1LU).
// CHECK:      ktdf_lowering.signal %[[L1SU]], %[[MNISU]]

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @signal_narrowing(%A: memref<?xf16, "DDR">,
                               %C: memref<?xf16, "DDR">) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    ktdf.pipeline {
      %l1_buf, %outer_t1, %outer_t2 = ktdf.private ->
          (memref<64xf16, "L1">, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, "L1">
        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %t1, %t2
          : memref<64xf16, "L1">, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%outer_t1) {
        ktdf.data_transfer from %A[%c0] size [64] to %l1_buf[%c0] size [64]
          : memref<?xf16, "DDR">,
            memref<64xf16, "L1">
      } {applicable_units = ["MNILU"]}
      ktdf.stage depends_in(%outer_t1)
                 depends_out(%outer_t2) {
        ktdf.pipeline {
          %inner_fifo1, %inner_fifo2, %inner_t1, %inner_t2 = ktdf.private ->
              (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
               !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
               !ktdf.token, !ktdf.token) {
            %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
            %f2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
            %t1 = ktdf.create_token : !ktdf.token
            %t2 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %f1, %f2, %t1, %t2
              : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
                !ktdf.token, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%inner_t1) {
            ktdf.data_transfer from %l1_buf[%c0] size [64] to %inner_fifo1 size [64]
              : memref<64xf16, "L1">,
                !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
          } {applicable_units = ["L1LU"]}
          ktdf.stage depends_in(%inner_t1)
                     depends_out(%inner_t2) {
            %v = ktdf.read_from_fifo %inner_fifo1
                   : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
            ktdf.write_to_fifo %v, %inner_fifo2
              : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
          } {applicable_units = ["SFU"]}
          ktdf.stage depends_in(%inner_t2) depends_out(none) {
            "test.store"(%inner_fifo2, %l1_buf) : (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
              memref<64xf16, "L1">) -> ()
          } {applicable_units = ["L1SU"]}
        }
      } {applicable_units = ["L1LU", "SFU", "L1SU"]}
      ktdf.stage depends_in(%outer_t2) depends_out(none) {
        ktdf.data_transfer from %l1_buf[%c0] size [64] to %C[%c0] size [64]
          : memref<64xf16, "L1">,
            memref<?xf16, "DDR">
      } {applicable_units = ["MNISU"]}
    }
    return
  }
}
