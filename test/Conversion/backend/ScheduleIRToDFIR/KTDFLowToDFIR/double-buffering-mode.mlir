// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s

// Tests verifying double-buffering mode dispatch:
//   BufferFirst (L1LU) — subi placed at yield; iter_arg init = offset_0 (128)
//   BufferLast  (MNILU) — subi placed at loop-body top; iter_arg init = offset_1 (98432)

// CHECK-LABEL: func.func private @mode_a
// CHECK:         %[[C98560_A:.*]] = arith.constant 98560 : index
// CHECK:         %[[C128_A:.*]] = arith.constant 128 : index
// CHECK:         dataflow.program_unit iter_arg : {{.*}} -> ({{.*}}) : {
// CHECK:           dataflow.program_unit iter_arg : {{.*}} -> ({{.*}}) : {
// Mode A: outer iter_arg initialized to offset_0 (128).
// CHECK:             scf.for {{.*}} iter_args(%[[OUTER_A:.*]] = %[[C128_A]]) -> (index)
// Mode A: inner iter_arg cascaded from outer iter_arg.
// CHECK:               scf.for {{.*}} iter_args(%[[INNER_A:.*]] = %[[OUTER_A]]) -> (index)
// Mode A: get_logical_memory_view uses the iter_arg directly.
// CHECK:                 dataflow.get_logical_memory_view {{.*}}, %[[INNER_A]] {{.*}}
// Mode A: test.use consumes the view.
// CHECK:                 "test.use"
// Mode A: subi uses the precomputed constant (not an addi).
// CHECK-NEXT:            %[[SUBI_A:.*]] = arith.subi %[[C98560_A]], %[[INNER_A]] : index
// CHECK-NEXT:            scf.yield %[[SUBI_A]] : index

// CHECK-LABEL: func.func private @mode_b
// CHECK:         %[[C98560_B:.*]] = arith.constant 98560 : index
// CHECK:         %[[C98432_B:.*]] = arith.constant 98432 : index
// CHECK:         dataflow.program_unit iter_arg : {{.*}} -> ({{.*}}) : {
// CHECK:           dataflow.program_unit iter_arg : {{.*}} -> ({{.*}}) : {
// Mode B: outer iter_arg initialized to offset_1 (98432).
// CHECK:             scf.for {{.*}} iter_args(%[[OUTER_B:.*]] = %[[C98432_B]]) -> (index)
// Mode B: inner iter_arg cascaded from outer iter_arg.
// CHECK:               scf.for {{.*}} iter_args(%[[INNER_B:.*]] = %[[OUTER_B]]) -> (index)
// Mode B: subi is the FIRST op in the inner loop body (loop-body top).
// CHECK-NEXT:            %[[SUBI_B:.*]] = arith.subi %[[C98560_B]], %[[INNER_B]] : index
// Mode B: get_logical_memory_view uses the subi result (not the iter_arg).
// CHECK-NEXT:            dataflow.get_logical_memory_view {{.*}}, %[[SUBI_B]] {{.*}}
// CHECK:                 "test.use"
// Mode B: yield carries the subi result.
// CHECK:                 scf.yield %[[SUBI_B]] : index

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  // Mode A: L1LU uses BufferFirst — subi at yield (end of loop body),
  // iter_arg initialized to offset_0 (the smaller offset, 128).
  func.func private @mode_a() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %c128 = arith.constant 128 : index
    %c98432 = arith.constant 98432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c64 = arith.constant 64 : index
    dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
      %view0 = dataflow.get_logical_memory_view %0, %c128 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      %view1 = dataflow.get_logical_memory_view %0, %c98432 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      scf.for %i = %c0 to %c12 step %c1 {
        scf.for %j = %c0 to %c64 step %c1 {
          %phase = ktdf.buffer_phase(%i, %j) {num_phases = 2 : i64} : index
          %sel = ktdf.select_memref %phase[%view0, %view1] : memref<12x1x64x64xf16>
          "test.use"(%sel) : (memref<12x1x64x64xf16>) -> ()
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    }
    return
  }

  // Mode B: MNILU uses BufferLast — subi at loop-body top (first op),
  // iter_arg initialized to offset_1 (the larger offset, 98432).
  func.func private @mode_b() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %c128 = arith.constant 128 : index
    %c98432 = arith.constant 98432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c64 = arith.constant 64 : index
    dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
      %view0 = dataflow.get_logical_memory_view %0, %c128 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      %view1 = dataflow.get_logical_memory_view %0, %c98432 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      scf.for %i = %c0 to %c12 step %c1 {
        scf.for %j = %c0 to %c64 step %c1 {
          %phase = ktdf.buffer_phase(%i, %j) {num_phases = 2 : i64} : index
          %sel = ktdf.select_memref %phase[%view0, %view1] : memref<12x1x64x64xf16>
          "test.use"(%sel) : (memref<12x1x64x64xf16>) -> ()
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    }
    return
  }
}
