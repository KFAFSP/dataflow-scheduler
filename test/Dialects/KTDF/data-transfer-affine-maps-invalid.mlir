// RUN: not scheduler %s 2>&1 | FileCheck %s

// Verifier: source_map result count must match source memref rank.
// CHECK: 'ktdf.data_transfer' op source memref has rank 2 but source_map has 3 results
func.func @bad_source_map_rank(%i: index) {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  "ktdf.data_transfer"(%A, %i, %fifo) <{
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0>,
    static_source_sizes = array<i64: 1, 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<(d0) -> (d0, d0, d0)>
  }> : (memref<128x128xf16, "DRAM">, index,
        !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>) -> ()
  return
}

// Verifier: source_map cannot have symbols.
// CHECK: 'ktdf.data_transfer' op source_map must have 0 symbols (dims-only); got 1
func.func @bad_source_map_symbols(%i: index, %n: index) {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  "ktdf.data_transfer"(%A, %i, %n, %fifo) <{
    operandSegmentSizes = array<i32: 1, 2, 0, 1, 0, 0>,
    static_source_sizes = array<i64: 1, 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<(d0)[s0] -> (d0 + s0, d0)>
  }> : (memref<128x128xf16, "DRAM">, index, index,
        !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>) -> ()
  return
}

// Verifier: source_map must be absent for fifo source.
// CHECK: 'ktdf.data_transfer' op source_map must be absent for fifo source
func.func @bad_fifo_with_map() {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  %lx = memref.alloc() : memref<64xf16, "LX">
  %c0 = arith.constant 0 : index
  "ktdf.data_transfer"(%fifo, %lx, %c0) <{
    operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 0>,
    static_source_sizes = array<i64: 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<() -> ()>,
    dest_map = affine_map<(d0) -> (d0)>
  }> : (!ktdf.fifo.slot<"hbm-to-sfp", 64xf16>, memref<64xf16, "LX">,
        index) -> ()
  return
}
