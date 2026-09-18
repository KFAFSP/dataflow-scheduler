// Path expansion of unrealizable FIFO types.
//
// A FIFO is realized by one link of the device, so its endpoints name the two
// ends of that link.  Both pipelines here ask for FIFOs that name a memory
// instead, which no link does, and each has to be legalized a different way:
//
//   - @invalid_fifo_ddr_to_sfu asks for <"DDR" -> "SFU">.  The only FIFO
//     reaching the SFU is the L1LU one fed from L1, so the transfer is routed
//     around it: a DDR -> L1 DMA (MNILU) into a private L1 buffer, followed by
//     the L1 -> SFU FIFO (L1LU).
//   - @invalid_fifo_l1_to_sfu asks for <"L1" -> "SFU"> and <"SFU" -> "L1"> with
//     the data already in L1.  There is no hop to insert; only the endpoints
//     move onto the units that do the moving (L1LU, L1SU), and every stage ends
//     up pinned to the unit it runs on.
//
// The store side of the first pipeline already names a realizable FIFO
// (<"SFU" -> "L1SU">), so its body must come out unchanged.

// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @invalid_fifo_ddr_to_sfu() {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_1]], sizes: [1, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<1x64xf16> to memref<1x64xf16, "DDR">
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:6 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:         %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[ALLOC_1]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]], %[[CREATE_TOKEN_2]] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#3) {
// CHECK-NEXT:         ktdf.data_transfer from %[[MEMORY_SPACE_CAST_0]][0, 0] size [1, 64] to %[[VAL_0]]#1[0, 0] size [1, 64] : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["MNILU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_1:.*]]#3) depends_out(%[[VAL_1]]#4) {
// CHECK-NEXT:         ktdf.data_transfer from %[[VAL_1]]#1[0, 0] size [1, 64] to %[[VAL_1]]#2 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["L1LU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_2:.*]]#4) depends_out(%[[VAL_2]]#5) {
// CHECK-NEXT:         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_2]]#2 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:         ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK-NEXT:           %[[NEGF_0:.*]] = arith.negf %[[VAL_3]] : f16
// CHECK-NEXT:           linalg.yield %[[NEGF_0]] : f16
// CHECK-NEXT:         } -> tensor<64xf16>
// CHECK-NEXT:         ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_2]]#0 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["SFU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_5:.*]]#5) depends_out(none) {
// CHECK-NEXT:         ktdf.data_transfer from %[[VAL_5]]#0 size [64] to %[[ALLOC_0]][0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["L1SU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @invalid_fifo_l1_to_sfu() {
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:     %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:         ktdf.data_transfer from %[[ALLOC_0]][0, 0] size [1, 64] to %[[VAL_0]]#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["L1LU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_1:.*]]#2) depends_out(%[[VAL_1]]#3) {
// CHECK-NEXT:         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_1]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<64xf16>) outs(%[[EMPTY_0]] : tensor<64xf16>) {
// CHECK-NEXT:         ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16):
// CHECK-NEXT:           %[[NEGF_0:.*]] = arith.negf %[[VAL_2]] : f16
// CHECK-NEXT:           linalg.yield %[[NEGF_0]] : f16
// CHECK-NEXT:         } -> tensor<64xf16>
// CHECK-NEXT:         ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_1]]#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:       } {applicable_units = ["SFU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_4:.*]]#3) depends_out(none) {
// CHECK-NEXT:         ktdf.data_transfer from %[[VAL_4]]#1 size [64] to %[[ALLOC_1]][0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["L1SU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @invalid_fifo_ddr_to_sfu() {
    %c1024 = arith.constant 1024 : index

    %src_mv = ktdp.construct_memory_view %c1024, sizes: [1, 64], strides: [64, 1]
        {coordinate_set = #set, memory_space = #ktdp.memory_space<global>}
        : memref<1x64xf16>
    %src = memref.memory_space_cast %src_mv
        : memref<1x64xf16> to memref<1x64xf16, "DDR">

    %dst = memref.alloc() : memref<1x64xf16, "L1">

    ktdf.pipeline {
      %prv:4 = ktdf.private -> (
          !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>,
          !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
          !ktdf.token,
          !ktdf.token
      ) {
        %fifo_ld = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
        %fifo_st = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %tok0 = ktdf.create_token : !ktdf.token
        %tok1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %fifo_ld, %fifo_st, %tok0, %tok1
            : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>,
              !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
              !ktdf.token, !ktdf.token
      }

      ktdf.stage depends_in(none) depends_out(%prv#2) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %prv#0 size [64]
            : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      }

      ktdf.stage depends_in(%prv#2) depends_out(%prv#3) {
        %in = ktdf.read_from_fifo %prv#0
            : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
        %init = tensor.empty() : tensor<64xf16>
        %out = linalg.generic {indexing_maps = [#map, #map],
                               iterator_types = ["parallel"]}
            ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
        ^bb0(%v: f16, %acc: f16):
          %neg = arith.negf %v : f16
          linalg.yield %neg : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %out, %prv#1
            : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      } {applicable_units = ["SFU"]}

      ktdf.stage depends_in(%prv#3) depends_out(none) {
        ktdf.data_transfer from %prv#1 size [64] to %dst[0, 0] size [1, 64]
            : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
      }
    }
    return
  }

  func.func @invalid_fifo_l1_to_sfu() {
    %src = memref.alloc() : memref<1x64xf16, "L1">
    %dst = memref.alloc() : memref<1x64xf16, "L1">

    ktdf.pipeline {
      %prv:4 = ktdf.private -> (
          !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>,
          !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>,
          !ktdf.token,
          !ktdf.token
      ) {
        %fifo_ld = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
        %fifo_st = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
        %tok0 = ktdf.create_token : !ktdf.token
        %tok1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %fifo_ld, %fifo_st, %tok0, %tok1
            : !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>,
              !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>,
              !ktdf.token, !ktdf.token
      }

      ktdf.stage depends_in(none) depends_out(%prv#2) {
        ktdf.data_transfer from %src[0, 0] size [1, 64] to %prv#0 size [64]
            : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
      }

      ktdf.stage depends_in(%prv#2) depends_out(%prv#3) {
        %in = ktdf.read_from_fifo %prv#0
            : !ktdf.fifo.slot<"L1" -> "SFU", 64xf16> -> tensor<64xf16>
        %init = tensor.empty() : tensor<64xf16>
        %out = linalg.generic {indexing_maps = [#map, #map],
                               iterator_types = ["parallel"]}
            ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
        ^bb0(%v: f16, %acc: f16):
          %neg = arith.negf %v : f16
          linalg.yield %neg : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %out, %prv#1
            : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
      } {applicable_units = ["SFU"]}

      ktdf.stage depends_in(%prv#3) depends_out(none) {
        ktdf.data_transfer from %prv#1 size [64] to %dst[0, 0] size [1, 64]
            : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<1x64xf16, "L1">
      }
    }
    return
  }
}
