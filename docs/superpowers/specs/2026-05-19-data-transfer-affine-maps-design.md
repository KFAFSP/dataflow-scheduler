# Design: Affine maps on `ktdf.data_transfer`

**Status:** Proposed
**Date:** 2026-05-19
**Branch:** `pc/pipelinefixes`
**Scope:** Refactor `KTDF_DataTransferOp` so each subscript can be an affine
expression of the index operands rather than a single SSA value. The op
gains two `OptionalAttr<AffineMapAttr>` fields (`source_map`, `dest_map`)
and implements `AffineMapAccessInterface`. Mirrors `affine.dma_start` /
`affine.load` syntactically. The producer in
`ConstructThreeStagePipeline.cpp` is simplified to pass linalg indexing
maps directly instead of materializing per-dim arith ops via
`affine::expandAffineMap`.

---

## 1. Scope & Goals

**Op.** `ktdf.data_transfer`. The verifier and parse/print are rewritten;
the operand layout is preserved. `MemoryEffectsOpInterface` is *not*
declared (decision matches the op's current state — explicitly excluded
per design discussion).

**Goal.** Carry per-subscript affine expressions inline in the IR so that

1. Broadcast/squeeze patterns (`(d0,d1,d2,d3,d4) -> (d0, d2, d3, 0)`)
   don't require materializing `arith.constant 0 : index` SSA values to
   feed the constant-zero subscript.
2. Loop-IV-projection patterns (`(d0,d1,d2,d3) -> (d1,d2,d3)`) don't
   require materializing arith ops via `affine::expandAffineMap` —
   `createDataTransfers` can pass the linalg indexing map directly to the
   op.
3. Future passes that reason about the access pattern can use
   `AffineMapAccessInterface::getAffineMapAttrForMemRef` to ask "what
   region of this memref does this transfer touch?" without
   re-deriving it from the operand list.

**Non-goals.**
- No `MemoryEffectsOpInterface` modeling. The op carries no memory
  effects (matches today's behavior).
- No symbols in the maps. Dims-only.
- No alternate non-affine syntax. Subscripts are *always* affine
  expressions. Plain `[%i, %j]` is itself the simplest affine form (it
  parses to the identity map `(d0,d1)->(d0,d1)` over operands
  `[%i, %j]`), so existing `.mlir` text round-trips unchanged.
- No support for fifo-side maps. Fifos have no subscripts; their map
  attribute is absent and `*_indices` is empty (existing rule).
- No striding or multi-level access patterns. The map's operand list and
  the size operands handle these orthogonally.

---

## 2. Op Shape (TableGen)

```tablegen
def KTDF_DataTransferOp : KTDF_Op<"data_transfer", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>
]> {
  let arguments = (ins
    AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$source,
    Variadic<Index>:$source_indices,
    Variadic<Index>:$source_sizes,
    AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$destination,
    Variadic<Index>:$dest_indices,
    Variadic<Index>:$dest_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$static_source_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$static_dest_sizes,
    OptionalAttr<AffineMapAttr>:$source_map,
    OptionalAttr<AffineMapAttr>:$dest_map
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}
```

`AttrSizedOperandSegments` is preserved. The two new map attrs are
`OptionalAttr` because each is present iff the corresponding side is a
memref.

---

## 3. Textual Form

### 3.1 Grammar

```
ktdf.data_transfer
  `from` <source-side>
  (`size` `[` $source_sizes `]` (with mixed static))?
  `to`   <dest-side>
  (`size` `[` $dest_sizes   `]`)?
  attr-dict `:` type($source) `,` type($destination)

<source-side> ::= $source ( `[` <affine-subscript-list> `]` )?
<dest-side>   ::= $destination ( `[` <affine-subscript-list> `]` )?

<affine-subscript-list> ::= comma-separated AffineExpr terms over the
                            inline SSA operands (dims-only)
```

Brackets are present iff that side is a memref.

### 3.2 Examples

Memref → fifo with non-trivial source subscript:

```mlir
ktdf.data_transfer
  from %A[%i + 3, %j * 2] size [1, 64]
  to %fifo size [64]
  : memref<128x128xf16, "DRAM">, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
```

Memref → memref with identity-shaped subscripts (still affine syntax;
the parser builds the identity map):

```mlir
ktdf.data_transfer
  from %A[%i, %j] size [1, 64]
  to %lx[%c0] size [64]
  : memref<128x128xf16, "DRAM">, memref<64xf16, "LX">
```

Fifo → memref with constant subscript (broadcast pattern from
`Softmax_225-Mul`):

```mlir
ktdf.data_transfer
  from %fifo size [64]
  to %B[%arg0, %arg2, %arg3, 0] size [1, 1, 1, 1]
  : !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>, memref<12x1x64x64xf16, ...>
```

Fifo → fifo is forbidden (existing rule, unchanged).

---

## 4. Parse / Print

Hand-rolled `parse(OpAsmParser&, OperationState&)` and
`print(OpAsmPrinter&)` in `KTDFOps.cpp`. Both directions use MLIR's
`OpAsmParser::parseAffineMapOfSSAIds` /
`OpAsmPrinter::printAffineMapOfSSAIds` helpers, which are exactly the
helpers `affine.load` and `affine.dma_start` use.

### 4.1 Parse flow

1. Parse `from` keyword.
2. Parse the source SSA value (as `OpAsmParser::UnresolvedOperand`).
3. If the next token is `[`, call `parseAffineMapOfSSAIds(<operands-out>,
   <attr-out>, "source_map", result.attributes)`. The helper consumes
   `[<expr>, <expr>, ...]`, builds an `AffineMap`, attaches it to
   `result.attributes` under name `"source_map"`, and writes the operand
   `UnresolvedOperand`s into the out-list. The driver then resolves
   those operands as `IndexType` and appends them to `result.operands` as
   `source_indices`.
4. Optional `size` clause — reuse the existing `SizesList` parse helper.
5. Parse `to` keyword.
6. Repeat steps 2–4 for the destination side, using attribute name
   `"dest_map"`.
7. Parse optional attr-dict and the `: type, type` suffix; resolve
   source and destination SSA values against their parsed types.
8. Compute and set `operandSegmentSizes` (required by
   `AttrSizedOperandSegments`).

### 4.2 Print flow

1. Emit `ktdf.data_transfer from `.
2. Print the source SSA value.
3. If `source_map` is present, call `printAffineMapOfSSAIds(p, op,
   source_indices, "source_map", attrs)` to emit `[<expr>, ...]`.
4. Optional `size` clause for source sizes (existing helper).
5. Emit ` to `, then the destination, then `dest_map` (same pattern),
   then dest sizes.
6. Print attr-dict, eliding `source_map`, `dest_map`,
   `static_source_sizes`, `static_dest_sizes`, and `operandSegmentSizes`
   (rendered inline).
7. Emit `: <source-type>, <dest-type>`.

---

## 5. Verifier

Replaces today's "indices count == rank" checks with map-aware checks,
applied per side:

1. **Source kind**: must be `MemRefType` or `FifoSlotType`. (Existing.)
2. **Destination kind**: must be `MemRefType` or `FifoSlotType`.
   (Existing.)
3. **No fifo-to-fifo**: source and destination cannot both be fifos.
   (Existing.)
4. **Element-type compatibility**: source and dest element types must
   match. (Existing.)
5. **Source map invariants** (new):
   - If source is a memref:
     - `source_map` attribute is present.
     - `source_map.getNumResults() == sourceMemRefType.getRank()`.
     - `source_map.getNumSymbols() == 0` (dims-only).
     - `source_indices.size() == source_map.getNumDims()`.
   - If source is a fifo:
     - `source_map` is absent.
     - `source_indices` is empty. (Existing rule, kept.)
6. **Destination map invariants** (new): same shape as rule 5, applied
   to `dest_map` / `dest_indices` / `destination`.
7. **Static/dynamic sizes consistency** (existing rule, kept verbatim):
   for each side, the count of dynamic placeholders inside
   `static_*_sizes` must equal the count of dynamic-size operands.

Error messages must be specific: `"source memref has rank 3 but
source_map has 4 results"`, `"source_map must have 0 symbols
(dims-only); got 2"`, `"source_map must be absent for fifo source"`, etc.

### 5.1 What the verifier does *not* check

- In-bounds of affine expressions against memref shape (would require
  range info we don't have).
- Whether size operands match the size implied by the access pattern
  (broadcast: size 1, but the map projects a dim — these are
  intentionally independent).
- Coordinate-set / access-tile-set semantics from
  `ktdp.construct_access_tile`. That belonged on the access tile op;
  `data_transfer` doesn't carry it.

---

## 6. AffineMapAccessInterface

Single method to provide:

```cpp
NamedAttribute DataTransferOp::getAffineMapAttrForMemRef(Value memref) {
  if (memref == getSource())
    return {StringAttr::get(getContext(), "source_map"),
            getSourceMapAttr()};
  assert(memref == getDestination() &&
         "memref must be source or destination");
  return {StringAttr::get(getContext(), "dest_map"), getDestMapAttr()};
}
```

If a caller passes a memref that's neither side, the assertion fires —
matches `affine.dma_start` precedent.

`getSourceMapAttr()` / `getDestMapAttr()` are auto-generated from the
`OptionalAttr<AffineMapAttr>` declarations and return
`AffineMapAttr` (or null when the attribute is absent for a fifo side).
Callers that use the interface are expected to only invoke it for
memref operands; this is the same precondition `affine.dma_start`
uses.

---

## 7. Builders

Two map-required overloads, mirroring the existing pair (one for mixed
static/dynamic sizes, one for all-static):

```cpp
// Mixed static/dynamic sizes via OpFoldResult.
OpBuilder<(ins
    "Value":$source,
    "AffineMap":$sourceMap,        // null iff source is a fifo
    "ValueRange":$sourceIndices,
    "ArrayRef<OpFoldResult>":$sourceSizes,
    "Value":$destination,
    "AffineMap":$destMap,          // null iff destination is a fifo
    "ValueRange":$destIndices,
    "ArrayRef<OpFoldResult>":$destSizes)>,

// All-static sizes.
OpBuilder<(ins
    "Value":$source,
    "AffineMap":$sourceMap,
    "ValueRange":$sourceIndices,
    "ArrayRef<int64_t>":$sourceSizes,
    "Value":$destination,
    "AffineMap":$destMap,
    "ValueRange":$destIndices,
    "ArrayRef<int64_t>":$destSizes)>
```

Builder body (same shape for both): pass through the operand lists and
sizes exactly as today, then attach `source_map` / `dest_map` iff the
corresponding `AffineMap` argument is non-null *and* the corresponding
side is a memref. Asserts (matching the verifier rules): map non-null
iff side is a memref.

`AffineMap()` is the null-map sentinel that C++ callers pass for fifo
sides.

No identity-map helper builder is exposed. Callers that don't have a
real map handy build one inline:

```cpp
auto identity =
    AffineMap::getMultiDimIdentityMap(memref.getRank(), &getContext());
```

---

## 8. Producer-Side Migration

### 8.1 `createDataTransfers` in `ConstructThreeStagePipeline.cpp`

**Before** (post our recent fixes), per load/store operand:
1. Look up the matching linalg `AffineMap`.
2. Call `projectIvsAndSizesThroughIndexingMap`, which:
   - Calls `affine::expandAffineMap` to materialize per-dim index
     `Value`s via arith ops.
   - Computes per-dim sizes from the map's result expressions
     (`AffineDimExpr` → tile size, `AffineConstantExpr` → 1).
3. Pass the materialized `Value`s to `DataTransferOp::create`.

**After**:
1. Look up the matching linalg `AffineMap`. (Unchanged.)
2. Compute *only* projected sizes (no index materialization). Size
   projection logic is unchanged: `AffineDimExpr` → `tile_sizes[k]`,
   `AffineConstantExpr` → 1, others → error.
3. Pass the linalg map directly as `sourceMap` (or `destMap`) and the
   raw `loop_ivs` as `sourceIndices` to `DataTransferOp::create`.

**Code that goes away**:
- `affine::expandAffineMap` call inside
  `projectIvsAndSizesThroughIndexingMap`.
- The function loses its `OpBuilder&`, `Location`, `loop_ivs`, and
  `projected_indices` arguments and is renamed to
  `projectSizesThroughIndexingMap`.
- `#include "mlir/IR/PatternMatch.h"` (no longer needs `IRRewriter` for
  affine expansion in this pass) — but
  `replaceAccessTilesWithReinterpretCast` still does, so the include
  stays if it uses it.

Net delta in `ConstructThreeStagePipeline.cpp`: ~30 lines deleted.

### 8.2 Other producer call sites

1. **`StageCoarsening/BufferExpansion.cpp:cloneDataTransfer`** (line
   209) and the `DataTransferOp::create` call at line 321: forwards the
   maps from the original op to the cloned one. Reads
   `getSourceMapAttr()` / `getDestMapAttr()` and passes them to the new
   builder.
2. **`BroadcastPromotion.cpp`**: uses `Operation::clone` which
   preserves attributes automatically. **No change.**
3. **`createDataTransfers`**: covered in 8.1.

### 8.3 Consumer call sites

These read `getSource()` / `getDestination()` /
`getSourceIndices()` / `getDestIndices()` / `getSourceSizes()`. After
the refactor, all four accessors return exactly the same operand
ranges as before. **No changes**:

- `src/Transform/DoubleBuffering.cpp`
- `src/Transform/ReusePromotion/Common/InvarianceCheck.cpp`
- `src/Transform/ReusePromotion/Common/WriteSetScan.cpp`
- `src/Transform/ReusePromotion/Common/CandidateSearch.{hpp,cpp}`
- `src/Utils/OperationUtils.cpp`

---

## 9. Test Strategy

### 9.1 New unit tests

`test/Dialects/KTDF/data-transfer-affine-maps.mlir` (new file).

Positive (round-trip parse + print):
- Memref → fifo with non-trivial source subscript: `[%i + 3, %j * 2]`.
- Fifo → memref with constant subscript (broadcast pattern from
  `Softmax_225-Mul`): `[%a, %b, %c, 0]`.
- Memref → memref where both sides have non-identity maps.
- Plain index lists `[%i, %j]` (verify the parser builds an identity
  map and the printer emits it back as `[%i, %j]`).

Negative (verifier errors):
- Source map results count ≠ source memref rank.
- Source map present on a fifo side.
- Source indices count ≠ source map dim count.
- Source map with symbols.
- Mirror of each above on the dest side.

### 9.2 Existing lit tests

All existing `.mlir` files round-trip without edits. "Explicit affine
syntax everywhere" means subscripts are *always* parsed as affine
expressions; plain `[%i, %j]` is the simplest valid affine form (the
identity map over those operands). The parser builds the identity map
and the printer emits the same plain form back, so no existing lit
test sees a textual diff. Verified by:

- `test/Dialects/KTDF/data-transfer-ops.mlir`
- `test/Transform/ComputeGroupExtraction/*.mlir`
- `test/Transform/DoubleBuffering/*.mlir`
- `test/Scheduler_pipeline/add_155_generic.mlir`

### 9.3 Regression bound

`ninja check-scheduler` baseline is 89/90 (pre-existing
`pipeline-test.mlir` failure unrelated to this work). Refactor must
not regress that.

---

## 10. Scope Summary

**Files modified**:
- `src/Dialects/KTDF/KTDFOps.td` — interface, attrs, custom format.
- `src/Dialects/KTDF/KTDFOps.cpp` — `parse`, `print`,
  `getAffineMapAttrForMemRef`, rewritten `verify`, updated builders.
- `src/Transform/ConstructThreeStagePipeline.cpp` — drop
  `affine::expandAffineMap` plumbing in
  `projectIvsAndSizesThroughIndexingMap`, rename, update
  `createDataTransfers` to pass maps.
- `src/Transform/StageCoarsening/BufferExpansion.cpp` — forward maps in
  `cloneDataTransfer`.

**Files added**:
- `test/Dialects/KTDF/data-transfer-affine-maps.mlir`.

**Files unchanged** (reviewed):
- `src/Transform/BroadcastPromotion.cpp`
- `src/Transform/DoubleBuffering.cpp`
- `src/Transform/ReusePromotion/Common/*.{hpp,cpp}`
- `src/Utils/OperationUtils.cpp`

**Estimated diff size**: ~250 lines net (≈+200 in `KTDFOps.{td,cpp}`,
≈-30 in `ConstructThreeStagePipeline.cpp`, ≈+80 new test).

---

## 11. What Becomes Natural After This Change

1. Carrying broadcast/squeeze constants
   (`(d0,d1,d2,d3,d4) -> (d0, d2, d3, 0)`) inline in the IR instead of
   materializing `arith.constant 0 : index` SSA values.
2. Future passes that reason about access patterns can use
   `AffineMapAccessInterface` to query the affine region a transfer
   touches without re-deriving it from operand lists.
3. Hand-authored LX-hop expansions can carry `%i + offset` style
   subscripts inline without synthetic `arith.addi`.

## 12. What Stays the Same

- Operand layout and `AttrSizedOperandSegments` discipline.
- Sizes handling (static and dynamic forms).
- All consumer code that walks `getSourceIndices()` /
  `getDestIndices()` / `getSource()` / `getDestination()`.
- All existing `.mlir` test files round-trip unchanged.
