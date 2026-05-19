# Affine maps on `ktdf.data_transfer` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `KTDF_DataTransferOp` so each subscript is an affine expression over the index operands, mirroring `affine.dma_start` / `affine.load`. Add `source_map` / `dest_map` `OptionalAttr<AffineMapAttr>` fields, implement `AffineMapAccessInterface`, swap to a hand-rolled custom assembly format, and migrate the producer in `ConstructThreeStagePipeline.cpp` to pass linalg indexing maps directly instead of materializing per-dim arith ops.

**Architecture:** The op gains two optional `AffineMapAttr` fields — present iff the corresponding side is a memref. The verifier enforces `map.getNumResults() == memref.rank`, `map.getNumSymbols() == 0`, and `indices.size() == map.getNumDims()`. Custom parse/print uses `OpAsmParser::parseAffineMapOfSSAIds` / `OpAsmPrinter::printAffineMapOfSSAIds`. The textual form for the trivial case (`[%i, %j]`) round-trips unchanged because the parser builds the identity map for it. C++ builders require an `AffineMap` per memref side (null for fifo sides). Producer-side migration drops `affine::expandAffineMap` plumbing.

**Tech Stack:** MLIR (LLVM 22.1.3), TableGen for op definitions, C++17, lit + FileCheck for tests. Build with Ninja (`cd build && ninja scheduler`), test with `ninja check-scheduler`.

---

## File Structure

**Modified:**
- `src/Dialects/KTDF/KTDFOps.td` — add `AffineMapAccessInterface` trait and the two `OptionalAttr<AffineMapAttr>` fields, switch to `hasCustomAssemblyFormat = 1`.
- `src/Dialects/KTDF/KTDFOps.cpp` — add `parse`, `print`, `getAffineMapAttrForMemRef`; rewrite `verify`; replace both builders with map-aware overloads.
- `src/Transform/ConstructThreeStagePipeline.cpp` — drop `affine::expandAffineMap` from `projectIvsAndSizesThroughIndexingMap`, rename to `projectSizesThroughIndexingMap`, update `createDataTransfers` to pass linalg indexing maps to the new builder.
- `src/Transform/StageCoarsening/BufferExpansion.cpp` — update `cloneDataTransfer` to extend each side's `AffineMap` with identity dims for the prepended buffer-expansion indices.

**Added:**
- `test/Dialects/KTDF/data-transfer-affine-maps.mlir` — round-trip + verifier-error coverage for affine subscripts.

**Reviewed but unchanged:**
- `src/Transform/BroadcastPromotion.cpp` — uses `Operation::clone`, preserves attributes automatically.
- `src/Transform/DoubleBuffering.cpp`, `src/Transform/ReusePromotion/Common/*`, `src/Utils/OperationUtils.cpp` — read source/dest indices and source/dest values; new attrs are invisible to them.

**Files-that-stay-untouched discipline:** Don't restructure `KTDFOps.cpp` — it has the StageOp/ParallelOp custom parsers we follow as a pattern, but they're not affected by this change.

---

## Build / Test Commands (used throughout)

```bash
# Always run from the build directory
cd /Users/prasanth-work/Documents/code/KTIR-Scheduler/build

# Incremental build of the scheduler binary
ninja scheduler

# Lit suite (89/90 baseline; only pipeline-test.mlir fails for unrelated reasons)
ninja check-scheduler

# Run a single lit test verbosely
/Users/prasanth-work/Documents/code/external/llvmorg-22.1.3/llvm/utils/lit/lit.py -v \
    /Users/prasanth-work/Documents/code/KTIR-Scheduler/build/test/Dialects/KTDF/data-transfer-affine-maps.mlir
```

`scheduler` with no transformation flags acts as a verify-only round-tripper:

```bash
/Users/prasanth-work/Documents/code/KTIR-Scheduler/build/scheduler/bin/scheduler /path/to/file.mlir
```

---

## Task 1: Add the new attribute fields and interface to TableGen

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.td:483-580`

The current op definition is at `src/Dialects/KTDF/KTDFOps.td:483`. We'll add `AffineMapAccessInterface` to the trait list, add two `OptionalAttr<AffineMapAttr>` fields at the end of `arguments`, and switch to `hasCustomAssemblyFormat = 1`. The two existing builder declarations are removed (we'll add map-aware ones in Task 6).

- [ ] **Step 1: Add the AffineOps include at the top of KTDFOps.td if missing**

Look at `src/Dialects/KTDF/KTDFOps.td` for existing `include` lines near the top. Add this if it's not already there (just after the existing includes):

```tablegen
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
```

This makes `AffineMapAccessInterface` available.

- [ ] **Step 2: Modify the op definition**

Replace lines 483-580 of `src/Dialects/KTDF/KTDFOps.td` with this exact block. The description and example sections inside `let description = [{ ... }]` stay the same; only the trait list, arguments, assembly format, and builders change.

```tablegen
def KTDF_DataTransferOp : KTDF_Op<"data_transfer", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>
  ]> {
  let summary = "Transfer data between memory locations and FIFO slots";
  let description = [{
    The `ktdf.data_transfer` operation represents explicit data movement in the
    dataflow pipeline. It supports three types of transfers:

    1. **memref to memref**: Transfer data between different memory spaces
       (e.g., DRAM to LX, LX to DRAM)
    2. **memref to fifo**: Transfer data from memory to a FIFO slot
       (e.g., LX to FIFO for streaming to compute units)
    3. **fifo to memref**: Transfer data from a FIFO slot to memory
       (e.g., FIFO from compute units to SFP registers or LX)

    Memref subscripts are affine expressions of the index operands, in the
    same form used by `affine.load` / `affine.dma_start`. Each side that is
    a memref carries an `AffineMapAttr` whose result count equals the
    memref's rank; the operand list feeds the map's dim inputs (no
    symbols).

    Fifo sides have no map and no indices.

    ## Examples

    ```mlir
    // Memref to fifo with non-trivial source subscript
    ktdf.data_transfer
      from %A[%i + 3, %j * 2] size [1, 64]
      to %fifo size [64]
      : memref<128x128xf16, "DRAM">, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>

    // Memref to memref with identity-shaped subscripts
    ktdf.data_transfer
      from %A[%i, %j] size [1, 64]
      to %lx[%c0] size [64]
      : memref<128x128xf16, "DRAM">, memref<64xf16, "LX">

    // Fifo to memref with constant subscript (broadcast pattern)
    ktdf.data_transfer
      from %fifo size [64]
      to %B[%arg0, %arg2, %arg3, 0] size [1, 1, 1, 1]
      : !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>, memref<12x1x64x64xf16, ...>
    ```
  }];

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

  let extraClassDeclaration = [{
    /// Returns true if the source is a memref
    bool isSourceMemRef() {
      return llvm::isa<mlir::MemRefType>(getSource().getType());
    }

    /// Returns true if the source is a FIFO slot
    bool isSourceFifo() {
      return llvm::isa<FifoSlotType>(getSource().getType());
    }

    /// Returns true if the destination is a memref
    bool isDestinationMemRef() {
      return llvm::isa<mlir::MemRefType>(getDestination().getType());
    }

    /// Returns true if the destination is a FIFO slot
    bool isDestinationFifo() {
      return llvm::isa<FifoSlotType>(getDestination().getType());
    }

    /// Returns the OpFoldResult vector of source sizes (mixed static/dynamic).
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSourceSizes();

    /// Returns the OpFoldResult vector of destination sizes (mixed static/dynamic).
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedDestSizes();
  }];
}
```

> **Note:** if the original op had additional `extraClassDeclaration` methods beyond `isSourceMemRef` / `isSourceFifo` / `getMixedSourceSizes`, preserve them. Read the original block at `KTDFOps.td:582` first and merge.

- [ ] **Step 3: Build to confirm TableGen and the .h.inc generator are happy**

```bash
cd /Users/prasanth-work/Documents/code/KTIR-Scheduler/build && ninja scheduler
```

Expected: build will fail in `KTDFOps.cpp` because the existing `verify()`, builders, and any code that references the now-removed `assemblyFormat` no longer compile. **That's expected** — we'll fix it in subsequent tasks. The TableGen step itself (`KTDFOps.h.inc`, `KTDFOps.cpp.inc`) must succeed without errors.

If TableGen errors out (look for `*.td:NN: error:`), it's a syntax issue in the new definition. Fix and retry.

- [ ] **Step 4: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.td
git commit -m "$(cat <<'EOF'
Add affine map attrs and AffineMapAccessInterface to ktdf.data_transfer

OptionalAttr<AffineMapAttr> source_map / dest_map; present iff that side
is a memref. Switch to hasCustomAssemblyFormat = 1 for hand-rolled
parse/print that handles the affine subscript form. Builders are
removed here and re-added in a follow-up commit.

Build is intentionally broken in KTDFOps.cpp until the parse/print and
builder updates land.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rewrite the verifier with map-aware checks

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.cpp:414-521` (the `DataTransferOp::verify()` body)

- [ ] **Step 1: Replace the verifier body**

Replace the entire `DataTransferOp::verify()` function (currently at `src/Dialects/KTDF/KTDFOps.cpp:414-521`) with this exact body:

```cpp
LogicalResult DataTransferOp::verify() {
  auto source_type = getSource().getType();
  auto dest_type = getDestination().getType();

  bool source_fifo = llvm::isa<FifoSlotType>(source_type);
  bool dest_fifo = llvm::isa<FifoSlotType>(dest_type);

  if (source_fifo && dest_fifo) {
    return emitOpError("cannot transfer directly from FIFO to FIFO");
  }

  bool source_memref = llvm::isa<MemRefType>(source_type);
  bool dest_memref = llvm::isa<MemRefType>(dest_type);

  if (!source_memref && !source_fifo) {
    return emitOpError("source must be a memref or FIFO slot");
  }
  if (!dest_memref && !dest_fifo) {
    return emitOpError("destination must be a memref or FIFO slot");
  }

  // Element-type compatibility.
  Type source_elem_type, dest_elem_type;
  if (source_memref) {
    source_elem_type = llvm::cast<MemRefType>(source_type).getElementType();
  } else {
    source_elem_type = llvm::cast<FifoSlotType>(source_type).getElementType();
  }
  if (dest_memref) {
    dest_elem_type = llvm::cast<MemRefType>(dest_type).getElementType();
  } else {
    dest_elem_type = llvm::cast<FifoSlotType>(dest_type).getElementType();
  }
  if (source_elem_type != dest_elem_type) {
    return emitOpError("source and destination element types must match, got ")
           << source_elem_type << " and " << dest_elem_type;
  }

  // Map / indices invariants per side.
  auto verifySide = [&](StringRef side_name, bool is_memref, Type ty,
                        std::optional<AffineMap> map,
                        OperandRange indices) -> LogicalResult {
    if (is_memref) {
      auto memref_ty = llvm::cast<MemRefType>(ty);
      if (!map) {
        return emitOpError(side_name)
               << "_map must be present when " << side_name << " is a memref";
      }
      if (map->getNumSymbols() != 0) {
        return emitOpError(side_name)
               << "_map must have 0 symbols (dims-only); got "
               << map->getNumSymbols();
      }
      if (map->getNumResults() != static_cast<unsigned>(memref_ty.getRank())) {
        return emitOpError(side_name)
               << " memref has rank " << memref_ty.getRank() << " but "
               << side_name << "_map has " << map->getNumResults()
               << " results";
      }
      if (indices.size() != map->getNumDims()) {
        return emitOpError(side_name)
               << "_indices count (" << indices.size() << ") must match "
               << side_name << "_map dim count (" << map->getNumDims() << ")";
      }
      return success();
    }
    // FIFO side
    if (map) {
      return emitOpError(side_name)
             << "_map must be absent for fifo " << side_name;
    }
    if (!indices.empty()) {
      return emitOpError("FIFO slot ")
             << side_name << " cannot have indices specified";
    }
    return success();
  };

  std::optional<AffineMap> source_map;
  if (auto src_map_attr = getSourceMapAttr())
    source_map = src_map_attr.getValue();
  std::optional<AffineMap> dest_map;
  if (auto dst_map_attr = getDestMapAttr())
    dest_map = dst_map_attr.getValue();

  if (failed(verifySide("source", source_memref, source_type, source_map,
                        getSourceIndices())))
    return failure();
  if (failed(verifySide("dest", dest_memref, dest_type, dest_map,
                        getDestIndices())))
    return failure();

  // Sizes consistency (dynamic operand count vs static array placeholders).
  auto verifySize = [&](OperandRange dynamicSizes,
                        std::optional<ArrayRef<int64_t>> static_sizes,
                        StringRef name) -> LogicalResult {
    if (!static_sizes) return success();
    ArrayRef<int64_t> static_values = *static_sizes;
    unsigned num_dynamic = llvm::count_if(
        static_values, [](int64_t val) { return ShapedType::isDynamic(val); });
    if (dynamicSizes.size() != num_dynamic) {
      return emitOpError("number of dynamic ")
             << name << " sizes (" << dynamicSizes.size()
             << ") must match number of dynamic entries in static_" << name
             << "_sizes (" << num_dynamic << ")";
    }
    return success();
  };
  if (failed(verifySize(getSourceSizes(), getStaticSourceSizes(), "source")))
    return failure();
  if (failed(verifySize(getDestSizes(), getStaticDestSizes(), "dest")))
    return failure();

  return success();
}
```

- [ ] **Step 2: Build (build will still fail elsewhere)**

```bash
ninja scheduler 2>&1 | grep -A 3 "KTDFOps.cpp.*error" | head -40
```

Expected: only errors should be in *other* parts of `KTDFOps.cpp` (parse/print/builders), not inside `verify()`. If you see compile errors inside `verify()`, fix them before moving on.

- [ ] **Step 3: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.cpp
git commit -m "$(cat <<'EOF'
Rewrite DataTransferOp::verify for map-aware invariants

Per side: if memref, source_map/dest_map must be present, dims-only,
result count == memref rank, indices count == map dim count. If fifo,
map must be absent, indices empty.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement custom parser and printer

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.cpp` — add `parse` and `print` to `DataTransferOp`.

The parse/print pair sits next to the `verify` we just wrote. We use `parseAffineMapOfSSAIds` / `printAffineMapOfSSAIds`, then thread the optional `size` clause via the existing `parseSizesList` / `printSizesList` helpers (already at `src/Dialects/KTDF/KTDFOps.cpp:59-121`).

- [ ] **Step 1: Add the `parse` method**

Insert this method *immediately after* `DataTransferOp::verify()` (so it's grouped with other DataTransferOp methods):

```cpp
ParseResult DataTransferOp::parse(OpAsmParser& parser, OperationState& result) {
  OpAsmParser::UnresolvedOperand source_operand, dest_operand;
  SmallVector<OpAsmParser::UnresolvedOperand> source_index_operands;
  SmallVector<OpAsmParser::UnresolvedOperand> source_size_operands;
  SmallVector<OpAsmParser::UnresolvedOperand> dest_index_operands;
  SmallVector<OpAsmParser::UnresolvedOperand> dest_size_operands;
  DenseI64ArrayAttr static_source_sizes_attr;
  DenseI64ArrayAttr static_dest_sizes_attr;
  Type source_type, dest_type;

  if (parser.parseKeyword("from") || parser.parseOperand(source_operand))
    return failure();

  // Source map + indices: present iff `[` follows.
  if (parser.getCurrentLocation().getPointer()[0] == '[') {
    Attribute source_map_attr;
    if (parser.parseAffineMapOfSSAIds(source_index_operands, source_map_attr,
                                      "source_map", result.attributes))
      return failure();
  }

  // Source size clause (optional): `size [ ... ]`.
  if (succeeded(parser.parseOptionalKeyword("size"))) {
    if (parseSizesList(parser, source_size_operands, static_source_sizes_attr))
      return failure();
    if (static_source_sizes_attr) {
      result.addAttribute("static_source_sizes", static_source_sizes_attr);
    }
  }

  if (parser.parseKeyword("to") || parser.parseOperand(dest_operand))
    return failure();

  if (parser.getCurrentLocation().getPointer()[0] == '[') {
    Attribute dest_map_attr;
    if (parser.parseAffineMapOfSSAIds(dest_index_operands, dest_map_attr,
                                      "dest_map", result.attributes))
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("size"))) {
    if (parseSizesList(parser, dest_size_operands, static_dest_sizes_attr))
      return failure();
    if (static_dest_sizes_attr) {
      result.addAttribute("static_dest_sizes", static_dest_sizes_attr);
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(source_type) ||
      parser.parseComma() || parser.parseType(dest_type))
    return failure();

  // Resolve operands.
  if (parser.resolveOperand(source_operand, source_type, result.operands))
    return failure();
  Type index_ty = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(source_index_operands, index_ty, result.operands))
    return failure();
  if (parser.resolveOperands(source_size_operands, index_ty, result.operands))
    return failure();
  if (parser.resolveOperand(dest_operand, dest_type, result.operands))
    return failure();
  if (parser.resolveOperands(dest_index_operands, index_ty, result.operands))
    return failure();
  if (parser.resolveOperands(dest_size_operands, index_ty, result.operands))
    return failure();

  // operandSegmentSizes: source(1), src_idx, src_sz, dest(1), dst_idx, dst_sz
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(source_index_operands.size()),
           static_cast<int32_t>(source_size_operands.size()), 1,
           static_cast<int32_t>(dest_index_operands.size()),
           static_cast<int32_t>(dest_size_operands.size())}));

  return success();
}
```

> **Note on `getCurrentLocation().getPointer()[0] == '['`:** this peek-ahead avoids a token-consumption gotcha in `OpAsmParser`. If a more idiomatic peek API is available in this LLVM version, prefer it. The existing `parseSizesList` peeks via `parseOptionalOperand` — that's a different shape.

- [ ] **Step 2: Add the `print` method**

Append immediately after `parse`:

```cpp
void DataTransferOp::print(OpAsmPrinter& p) {
  p << " from " << getSource();
  if (auto src_map = getSourceMapAttr()) {
    p.printAffineMapOfSSAIds(src_map, getSourceIndices());
  }
  if (!getSourceSizes().empty() || getStaticSourceSizes()) {
    p << " size ";
    printSizesList(p, getOperation(), getSourceSizes(),
                   getStaticSourceSizesAttr());
  }
  p << " to " << getDestination();
  if (auto dst_map = getDestMapAttr()) {
    p.printAffineMapOfSSAIds(dst_map, getDestIndices());
  }
  if (!getDestSizes().empty() || getStaticDestSizes()) {
    p << " size ";
    printSizesList(p, getOperation(), getDestSizes(), getStaticDestSizesAttr());
  }
  // Elide all attributes we render inline.
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"source_map", "dest_map", "static_source_sizes",
                       "static_dest_sizes", "operandSegmentSizes"});
  p << " : " << getSource().getType() << ", " << getDestination().getType();
}
```

- [ ] **Step 3: Build**

```bash
ninja scheduler 2>&1 | tail -20
```

Expected: builders are still broken (Task 6 fixes them), but parse/print should compile. If `parse` errors on `getCurrentLocation()`, replace that peek with the equivalent `parser.getNextToken().is(Token::l_square)` if the token API is exposed, or use `parser.parseOptionalLSquare()` followed by manual rewinding of state. The MLIR upstream `affine.dma_start` uses `parser.parseOptionalLSquare()` followed by the affine-map helper *with the bracket already consumed* — see `mlir/lib/Dialect/Affine/IR/AffineOps.cpp:1959`. If our peek approach is rejected, fall back to that pattern by passing `Delimiter::None` to the affine helper and pre-parsing `[` ... `]` ourselves.

- [ ] **Step 4: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.cpp
git commit -m "$(cat <<'EOF'
Implement DataTransferOp custom parse and print

Use parseAffineMapOfSSAIds / printAffineMapOfSSAIds for the bracketed
subscript clause. Reuse the existing parseSizesList / printSizesList
helpers for the size clauses. operandSegmentSizes is computed and
attached during parse.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `getAffineMapAttrForMemRef`

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.cpp` — add the interface method.

- [ ] **Step 1: Add the method**

Insert this method right after `DataTransferOp::print`:

```cpp
NamedAttribute DataTransferOp::getAffineMapAttrForMemRef(Value memref) {
  if (memref == getSource()) {
    return {StringAttr::get(getContext(), "source_map"), getSourceMapAttr()};
  }
  assert(memref == getDestination() &&
         "memref must be either source or destination of this op");
  return {StringAttr::get(getContext(), "dest_map"), getDestMapAttr()};
}
```

- [ ] **Step 2: Build (still expected to fail in builders)**

```bash
ninja scheduler 2>&1 | grep -A 1 "DataTransferOp::getAffineMapAttrForMemRef\|KTDFOps.cpp.*error" | head -20
```

Expected: no errors mentioning `getAffineMapAttrForMemRef`. Errors elsewhere in `KTDFOps.cpp` (builders) are still expected.

- [ ] **Step 3: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.cpp
git commit -m "$(cat <<'EOF'
Implement AffineMapAccessInterface for DataTransferOp

Single method getAffineMapAttrForMemRef returning source_map or
dest_map depending on which memref operand the caller passes. Asserts
the memref is one of the two op operands, matching affine.dma_start
precedent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Replace builders with map-aware overloads

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.cpp:609-654` (the two existing `DataTransferOp::build` definitions).

- [ ] **Step 1: Replace the two builders**

Replace lines 609-654 (the `// Build with mixed static/dynamic sizes` and `// Build with all static sizes` builders) with:

```cpp
// Build with mixed static/dynamic sizes (OpFoldResult).
// `sourceMap` must be non-null iff source is a memref; same for `destMap`.
void DataTransferOp::build(OpBuilder& builder, OperationState& result,
                           Value source, AffineMap sourceMap,
                           ValueRange source_indices,
                           ArrayRef<OpFoldResult> source_sizes,
                           Value destination, AffineMap destMap,
                           ValueRange dest_indices,
                           ArrayRef<OpFoldResult> dest_sizes) {
  SmallVector<int64_t> static_source_sizes, static_dest_sizes;
  SmallVector<Value> dynamic_source_sizes, dynamic_dest_sizes;
  dispatchIndexOpFoldResults(source_sizes, dynamic_source_sizes,
                             static_source_sizes);
  dispatchIndexOpFoldResults(dest_sizes, dynamic_dest_sizes, static_dest_sizes);

  // Source map: non-null iff source is a memref (asserted at verifier time).
  bool source_is_memref = llvm::isa<MemRefType>(source.getType());
  assert(static_cast<bool>(sourceMap) == source_is_memref &&
         "sourceMap must be non-null iff source is a memref");
  bool dest_is_memref = llvm::isa<MemRefType>(destination.getType());
  assert(static_cast<bool>(destMap) == dest_is_memref &&
         "destMap must be non-null iff destination is a memref");

  build(builder, result, source, source_indices, dynamic_source_sizes,
        destination, dest_indices, dynamic_dest_sizes,
        builder.getDenseI64ArrayAttr(static_source_sizes),
        builder.getDenseI64ArrayAttr(static_dest_sizes),
        sourceMap ? AffineMapAttr::get(sourceMap) : AffineMapAttr(),
        destMap ? AffineMapAttr::get(destMap) : AffineMapAttr());
}

// Build with all static sizes.
void DataTransferOp::build(OpBuilder& builder, OperationState& result,
                           Value source, AffineMap sourceMap,
                           ValueRange source_indices,
                           ArrayRef<int64_t> source_sizes, Value destination,
                           AffineMap destMap, ValueRange dest_indices,
                           ArrayRef<int64_t> dest_sizes) {
  SmallVector<OpFoldResult> source_size_values = llvm::map_to_vector<4>(
      source_sizes,
      [&](int64_t v) -> OpFoldResult { return builder.getI64IntegerAttr(v); });
  SmallVector<OpFoldResult> dest_size_values = llvm::map_to_vector<4>(
      dest_sizes,
      [&](int64_t v) -> OpFoldResult { return builder.getI64IntegerAttr(v); });

  build(builder, result, source, sourceMap, source_indices, source_size_values,
        destination, destMap, dest_indices, dest_size_values);
}
```

- [ ] **Step 2: Add the matching declarations to `KTDFOps.td`**

Replace the `let builders = [...]` block on the op (currently inside `KTDFOps.td:564-580`, removed by Task 1's rewrite — but TableGen needs them re-added so the C++ `build` overloads are visible) with:

```tablegen
  let builders = [
    OpBuilder<(ins "Value":$source, "AffineMap":$sourceMap,
      "ValueRange":$sourceIndices,
      "ArrayRef<OpFoldResult>":$sourceSizes,
      "Value":$destination, "AffineMap":$destMap,
      "ValueRange":$destIndices,
      "ArrayRef<OpFoldResult>":$destSizes)>,
    OpBuilder<(ins "Value":$source, "AffineMap":$sourceMap,
      "ValueRange":$sourceIndices,
      "ArrayRef<int64_t>":$sourceSizes,
      "Value":$destination, "AffineMap":$destMap,
      "ValueRange":$destIndices,
      "ArrayRef<int64_t>":$destSizes)>
  ];
```

Place this block inside the op definition right before `let extraClassDeclaration = ...` so the generated header declares both overloads.

- [ ] **Step 3: Implement `getMixedSourceSizes` / `getMixedDestSizes`**

Append to the bottom of `KTDFOps.cpp` (just before the next op section), if not already present from the previous code:

```cpp
SmallVector<OpFoldResult> DataTransferOp::getMixedSourceSizes() {
  return getMixedValues(getStaticSourceSizes().value_or(ArrayRef<int64_t>{}),
                        getSourceSizes(), getOperation()->getContext());
}

SmallVector<OpFoldResult> DataTransferOp::getMixedDestSizes() {
  return getMixedValues(getStaticDestSizes().value_or(ArrayRef<int64_t>{}),
                        getDestSizes(), getOperation()->getContext());
}
```

`getMixedValues` lives in `mlir/Dialect/Utils/StaticValueUtils.h`. Add `#include "mlir/Dialect/Utils/StaticValueUtils.h"` at the top of `KTDFOps.cpp` if not already present.

> **If these methods already existed before Task 1's rewrite stripped them**, just preserve the originals — read `git diff HEAD~3 src/Dialects/KTDF/KTDFOps.cpp` to find them.

- [ ] **Step 4: Build**

```bash
ninja scheduler 2>&1 | tail -10
```

Expected: success now that builders, parse, print, and verify all compile. If linker errors mention undefined `getMixedSourceSizes`/`getMixedDestSizes`, the auto-generated declarations in `extraClassDeclaration` are looking for them; ensure Step 3 actually wrote them.

- [ ] **Step 5: Round-trip a hand-written op to confirm parse + print + verify all work**

Save this to `/tmp/data_transfer_smoke.mlir`:

```mlir
module {
  func.func @smoke() {
    %A = memref.alloc() : memref<128x128xf16, "DRAM">
    %lx = memref.alloc() : memref<64xf16, "LX">
    %c0 = arith.constant 0 : index
    %i = arith.constant 1 : index
    %j = arith.constant 2 : index
    ktdf.data_transfer
      from %A[%i + 3, %j * 2] size [1, 64]
      to %lx[%c0] size [64]
      : memref<128x128xf16, "DRAM">, memref<64xf16, "LX">
    return
  }
}
```

Run it:

```bash
/Users/prasanth-work/Documents/code/KTIR-Scheduler/build/scheduler/bin/scheduler /tmp/data_transfer_smoke.mlir
```

Expected: prints back the same IR (with normalized formatting). Exit code 0. If it crashes or errors, the parse, print, or verify has a bug — fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.td src/Dialects/KTDF/KTDFOps.cpp
git commit -m "$(cat <<'EOF'
Replace DataTransferOp builders with map-aware overloads

Each builder takes an AffineMap per memref side; null sentinel for the
fifo side. Asserts (matching the verifier rules) catch null-vs-memref
mismatches at construction time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Migrate `ConstructThreeStagePipeline.cpp` producer

**Files:**
- Modify: `src/Transform/ConstructThreeStagePipeline.cpp:726-776` (`projectIvsAndSizesThroughIndexingMap`).
- Modify: `src/Transform/ConstructThreeStagePipeline.cpp:778-869` (`createDataTransfers`).

The current helper materializes per-dim arith ops via `affine::expandAffineMap` and returns both indices and sizes. After the refactor, the helper computes only sizes; indices are no longer needed (the linalg map is passed directly to the op).

- [ ] **Step 1: Slim `projectIvsAndSizesThroughIndexingMap` to compute only sizes**

Replace `projectIvsAndSizesThroughIndexingMap` (currently at `src/Transform/ConstructThreeStagePipeline.cpp:734-776`) with:

```cpp
// Project tile_sizes through indexing_map to produce one entry per result
// dim of the map. Sizes are derived per result expr:
//   - AffineDimExpr  d_k  -> tile_sizes[k]   (projection of one iterator dim)
//   - AffineConstantExpr -> 1                (broadcast/squeeze: that source
//                                             dim is accessed at a fixed index)
// Other expressions (e.g. d0 + d1) are not currently supported and produce
// an error.
static LogicalResult projectSizesThroughIndexingMap(
    AffineMap indexing_map, ArrayRef<int64_t> tile_sizes,
    SmallVectorImpl<int64_t>& projected_sizes,
    function_ref<InFlightDiagnostic()> emit_error) {
  assert(indexing_map.getNumSymbols() == 0 &&
         "linalg indexing maps are expected to have no symbols");
  assert(indexing_map.getNumDims() == tile_sizes.size() &&
         "indexing map dims must match the iteration-space rank");

  for (AffineExpr result : indexing_map.getResults()) {
    if (auto dim = dyn_cast<AffineDimExpr>(result)) {
      projected_sizes.push_back(tile_sizes[dim.getPosition()]);
      continue;
    }
    if (isa<AffineConstantExpr>(result)) {
      projected_sizes.push_back(1);
      continue;
    }
    std::string buf;
    llvm::raw_string_ostream os(buf);
    result.print(os);
    emit_error() << "operand indexing map has an unsupported result expression ("
                 << buf
                 << "); only single-dim projections and constant indices are "
                    "supported";
    return failure();
  }
  return success();
}
```

- [ ] **Step 2: Update `createDataTransfers` to call the new helper and pass maps**

Replace lines 842-867 of `src/Transform/ConstructThreeStagePipeline.cpp` (the block from `// Project loop IVs and tile_sizes through the operand's indexing map` through the end of the for-body) with:

```cpp
    // Project tile_sizes through the operand's indexing map.
    SmallVector<int64_t> access_tile_sizes;
    if (failed(projectSizesThroughIndexingMap(
            *indexing_map, tile_sizes, access_tile_sizes,
            [&]() { return err_anchor->emitError(); }))) {
      signalPassFailure();
      return;
    }

    // Get the FIFO slot from ktdf.private results.
    Value fifo_slot = private_op.getResult(private_result_offset + i);

    // The fifo side gets a null AffineMap; the access-tile (memref) side
    // gets the linalg operand's indexing map directly. loop_ivs feeds the
    // map's dim inputs.
    AffineMap null_map;
    if (is_load) {
      mlir::ktdf::DataTransferOp::create(
          builder, loc, access_tile_value, *indexing_map, loop_ivs,
          access_tile_sizes, fifo_slot, null_map, ValueRange{}, fifo_sizes);
    } else {
      mlir::ktdf::DataTransferOp::create(
          builder, loc, fifo_slot, null_map, ValueRange{}, fifo_sizes,
          access_tile_value, *indexing_map, loop_ivs, access_tile_sizes);
    }
  }
}
```

- [ ] **Step 3: Drop unused includes**

In `src/Transform/ConstructThreeStagePipeline.cpp`, check whether anything else in the file uses `affine::expandAffineMap`. The `replaceAccessTilesWithReinterpretCast` function at line ~889 still does — leave it alone. So the `#include "mlir/Dialect/Affine/Utils.h"` stays. No include-line edits needed.

- [ ] **Step 4: Build**

```bash
ninja scheduler 2>&1 | tail -10
```

Expected: success. If `loop_ivs` is the wrong type (`SmallVector<Value>` vs `ValueRange`), the `DataTransferOp::create` overload resolution will fail — convert with `ValueRange{loop_ivs}` if needed.

- [ ] **Step 5: Run the full lit suite**

```bash
ninja check-scheduler 2>&1 | tail -15
```

Expected: 89 passed / 1 failed (`pipeline-test.mlir` is the pre-existing failure). If any other test fails, inspect with `-v` to see what changed:

```bash
/Users/prasanth-work/Documents/code/external/llvmorg-22.1.3/llvm/utils/lit/lit.py -v \
    /Users/prasanth-work/Documents/code/KTIR-Scheduler/build/test/Transform/ComputeGroupExtraction/basic.mlir
```

Most likely failure cause: a CHECK line that pinned the old textual form (with materialized `arith.muli` ops in the index slot). The new IR has the affine map carry the multiplication, eliminating the arith op. If that's the case, regenerate the CHECKs the same way we did earlier (via `mlir/utils/generate-test-checks.py` then post-edit any `CHECK-LABEL: module {` lines).

- [ ] **Step 6: Verify on the historically-failing inputs**

```bash
/Users/prasanth-work/Documents/code/KTIR-Scheduler/build/scheduler/bin/scheduler --kEmitDFIR \
    /Users/prasanth-work/Documents/code/KTIR-Scheduler/test/Scheduler_pipeline/add_155_generic.mlir 2>&1 | tail -20

/Users/prasanth-work/Documents/code/KTIR-Scheduler/build/scheduler/bin/scheduler \
    --compute-group-extraction --construct-three-stage-pipeline \
    /Users/prasanth-work/Documents/code/ktir-sdsc-translator/tests/sdsc/Softmax_225-Mul/ktir.mlir 2>&1 | tail -10
```

Both should succeed through `ConstructThreeStagePipelinePass` (failing later in `ParallelizeLoopsAcrossInstancesPass` — that's the pre-existing crash, unrelated). The expected post-pass IR for the broadcast operand should now read like:

```mlir
ktdf.data_transfer
  from %fifo size [64]
  to %B[%arg0, %arg2, %arg3, 0] size [1, 1, 1, 1]
  : !ktdf.fifo.slot<...>, memref<12x1x64x64xf16, ...>
```

— the `0` is now an `AffineConstantExpr` carried by `dest_map`, not a synthetic `%c0_23 = arith.constant 0 : index`.

- [ ] **Step 7: Commit**

```bash
git add src/Transform/ConstructThreeStagePipeline.cpp
git commit -m "$(cat <<'EOF'
Pass linalg indexing maps directly to ktdf.data_transfer

createDataTransfers no longer materializes per-dim arith ops via
affine::expandAffineMap. The slimmed helper projectSizesThroughIndexingMap
returns only the per-dim sizes; the loop IVs are passed straight to the
new map-aware DataTransferOp builder, which carries the linalg
AffineMap as source_map (or dest_map). Constant subscripts in the
indexing map (broadcast/squeeze cases) now appear inline in the IR
instead of as synthetic arith.constant SSA values.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Migrate `BufferExpansion::cloneDataTransfer`

**Files:**
- Modify: `src/Transform/StageCoarsening/BufferExpansion.cpp:209-326` (`cloneDataTransfer`).

The current code prepends "expanded" loop-IV-derived indices in front of the original indices. Under the new design, the `AffineMap` for that side must also extend its result list by N identity dims at the front (where N is the count of prepended indices), and its dim count grows by N.

- [ ] **Step 1: Add a helper to extend an AffineMap with leading identity dims**

Insert this static helper at the top of `src/Transform/StageCoarsening/BufferExpansion.cpp` (after the existing includes, before any function definitions):

```cpp
// Given an AffineMap `original` of shape (d0..d_{m-1}) -> (e0, ..., e_{r-1}),
// produce a new map (d0..d_{m+n-1}) -> (d0, d1, ..., d_{n-1}, e0', ..., e_{r-1}')
// where each e_i' is e_i with its dim positions shifted by +n.
static mlir::AffineMap prependIdentityDims(mlir::AffineMap original,
                                           unsigned n,
                                           mlir::MLIRContext* ctx) {
  if (n == 0) return original;
  unsigned old_dims = original.getNumDims();
  unsigned new_dims = old_dims + n;

  llvm::SmallVector<mlir::AffineExpr> new_results;
  new_results.reserve(n + original.getNumResults());

  // Leading identity dims for the prepended indices.
  for (unsigned i = 0; i < n; ++i) {
    new_results.push_back(mlir::getAffineDimExpr(i, ctx));
  }

  // Original results, shifted up by n.
  for (mlir::AffineExpr e : original.getResults()) {
    new_results.push_back(e.shiftDims(/*shift=*/n));
  }

  return mlir::AffineMap::get(new_dims, /*symbolCount=*/0, new_results, ctx);
}
```

- [ ] **Step 2: Read original maps, extend, and pass to the new builder**

Modify `cloneDataTransfer` (`src/Transform/StageCoarsening/BufferExpansion.cpp:209`). Right before the `auto new_transfer = mlir::ktdf::DataTransferOp::create(...)` call at line 321, insert:

```cpp
  // Extend the source/dest affine maps with identity dims for the
  // expansion-prepended indices, in lockstep with new_src_indices /
  // new_dst_indices growth above.
  unsigned src_prepend = src_info ? src_info->loop_ivs.size() : 0;
  unsigned dst_prepend = dst_info ? dst_info->loop_ivs.size() : 0;
  mlir::MLIRContext* ctx = transfer_op->getContext();

  mlir::AffineMap new_src_map;
  if (transfer_op.isSourceMemRef()) {
    mlir::AffineMap orig = transfer_op.getSourceMapAttr().getValue();
    new_src_map = prependIdentityDims(orig, src_prepend, ctx);
  }
  mlir::AffineMap new_dst_map;
  if (transfer_op.isDestinationMemRef()) {
    mlir::AffineMap orig = transfer_op.getDestMapAttr().getValue();
    new_dst_map = prependIdentityDims(orig, dst_prepend, ctx);
  }
```

- [ ] **Step 3: Update the `DataTransferOp::create` call to pass the maps**

Replace the existing `auto new_transfer = mlir::ktdf::DataTransferOp::create(...)` block at line 321-323 with:

```cpp
  auto new_transfer = mlir::ktdf::DataTransferOp::create(
      builder, transfer_op.getLoc(), src, new_src_map, new_src_indices,
      src_sizes, dst, new_dst_map, new_dst_indices, dst_sizes);
```

- [ ] **Step 4: Build**

```bash
ninja scheduler 2>&1 | tail -10
```

Expected: success.

- [ ] **Step 5: Run the StageCoarsening lit tests**

```bash
/Users/prasanth-work/Documents/code/external/llvmorg-22.1.3/llvm/utils/lit/lit.py -v \
    /Users/prasanth-work/Documents/code/KTIR-Scheduler/build/test/Transform/StageCoarsening/
```

Expected: all StageCoarsening tests pass. If a test fails because its CHECK lines pinned the old textual form, the new IR's affine map will inline the prepended identity dims (e.g. instead of `[%phase, %i, %j]` you'll see the same `[%phase, %i, %j]` because the parser still emits identity-form subscripts the same way). So failures are unlikely *unless* the test's CHECK was sensitive to the absence of an attribute that's now present. Inspect and update if needed.

- [ ] **Step 6: Run the full lit suite**

```bash
ninja check-scheduler 2>&1 | tail -10
```

Expected: 89 passed / 1 failed (the pre-existing `pipeline-test.mlir`).

- [ ] **Step 7: Commit**

```bash
git add src/Transform/StageCoarsening/BufferExpansion.cpp
git commit -m "$(cat <<'EOF'
Extend AffineMap when prepending indices in BufferExpansion clone

When cloneDataTransfer prepends N expansion-derived indices onto a
data_transfer's source/dest indices, the corresponding side's
AffineMap must grow by N leading identity dims so the verifier rule
"map.getNumResults() == memref.rank" continues to hold for the cloned
op (whose memref's rank also grew by N).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add the new lit test

**Files:**
- Create: `test/Dialects/KTDF/data-transfer-affine-maps.mlir`

- [ ] **Step 1: Write the lit test file**

Save this content to `test/Dialects/KTDF/data-transfer-affine-maps.mlir`:

```mlir
// RUN: scheduler %s | scheduler | FileCheck %s
// RUN: scheduler --split-input-file --verify-diagnostics %s

// Round-trip: memref-to-fifo with non-identity source subscript.
// CHECK-LABEL: func.func @transfer_with_affine_subscript
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}}[%{{.*}} + 3, %{{.*}} * 2]
// CHECK-SAME: size [1, 64]
// CHECK-SAME: to %{{.*}} size [64]
func.func @transfer_with_affine_subscript() {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  %i = arith.constant 1 : index
  %j = arith.constant 2 : index
  ktdf.data_transfer
    from %A[%i + 3, %j * 2] size [1, 64]
    to %fifo size [64]
    : memref<128x128xf16, "DRAM">, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  return
}

// -----

// Round-trip: fifo-to-memref with constant subscript (broadcast pattern).
// CHECK-LABEL: func.func @transfer_with_constant_subscript
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}} size [64]
// CHECK-SAME: to %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, 0]
// CHECK-SAME: size [1, 1, 1, 1]
func.func @transfer_with_constant_subscript(%a: index, %b: index, %c: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>
  %B = memref.alloc() : memref<12x1x64x64xf16, "DRAM">
  ktdf.data_transfer
    from %fifo size [64]
    to %B[%a, %b, %c, 0] size [1, 1, 1, 1]
    : !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>, memref<12x1x64x64xf16, "DRAM">
  return
}

// -----

// Round-trip: identity-form subscripts (existing pattern, preserved).
// CHECK-LABEL: func.func @transfer_identity
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}}[%{{.*}}, %{{.*}}] size [1, 64]
// CHECK-SAME: to %{{.*}}[%{{.*}}] size [64]
func.func @transfer_identity(%i: index, %j: index, %c0: index) {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %lx = memref.alloc() : memref<64xf16, "LX">
  ktdf.data_transfer
    from %A[%i, %j] size [1, 64]
    to %lx[%c0] size [64]
    : memref<128x128xf16, "DRAM">, memref<64xf16, "LX">
  return
}

// -----

// Verifier: source_map result count must match source memref rank.
func.func @bad_source_map_rank(%i: index) {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  // expected-error @+1 {{source memref has rank 2 but source_map has 3 results}}
  "ktdf.data_transfer"(%A, %i, %fifo) <{
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0>,
    static_source_sizes = array<i64: 1, 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<(d0) -> (d0, d0, d0)>
  }> : (memref<128x128xf16, "DRAM">, index,
        !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>) -> ()
  return
}

// -----

// Verifier: source_map cannot have symbols.
func.func @bad_source_map_symbols(%i: index, %n: index) {
  %A = memref.alloc() : memref<128x128xf16, "DRAM">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  // expected-error @+1 {{source_map must have 0 symbols (dims-only); got 1}}
  "ktdf.data_transfer"(%A, %i, %n, %fifo) <{
    operandSegmentSizes = array<i32: 1, 2, 0, 1, 0, 0>,
    static_source_sizes = array<i64: 1, 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<(d0)[s0] -> (d0 + s0, d0)>
  }> : (memref<128x128xf16, "DRAM">, index, index,
        !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>) -> ()
  return
}

// -----

// Verifier: source_map must be absent for fifo source.
func.func @bad_fifo_with_map() {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
  %lx = memref.alloc() : memref<64xf16, "LX">
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{source_map must be absent for fifo source}}
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
```

> **Note:** the negative-test stanzas use the generic `"ktdf.data_transfer"(...)` form so they bypass the custom parser and exercise just the verifier. This is a common MLIR pattern for testing verifier-only failure paths.

- [ ] **Step 2: Run the new test**

```bash
/Users/prasanth-work/Documents/code/external/llvmorg-22.1.3/llvm/utils/lit/lit.py -v \
    /Users/prasanth-work/Documents/code/KTIR-Scheduler/build/test/Dialects/KTDF/data-transfer-affine-maps.mlir
```

Expected: PASS. If a CHECK doesn't match, inspect the actual printer output (run `scheduler /path/to/file.mlir` directly) and tighten the CHECK lines.

- [ ] **Step 3: Run the full lit suite**

```bash
ninja check-scheduler 2>&1 | tail -10
```

Expected: 90 passed / 1 failed (the new test plus the previous 89; only `pipeline-test.mlir` fails for unrelated reasons).

- [ ] **Step 4: Commit**

```bash
git add test/Dialects/KTDF/data-transfer-affine-maps.mlir
git commit -m "$(cat <<'EOF'
Add lit test for ktdf.data_transfer affine subscripts

Round-trip coverage for non-identity source subscript, constant subscript
(broadcast/squeeze), and identity-form subscripts. Negative-path
verifier tests exercise: result count mismatch with memref rank,
symbols rejection, and map-on-fifo-side rejection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final integration check

**Files:** none modified. Smoke run + housekeeping.

- [ ] **Step 1: Full clean build**

```bash
cd /Users/prasanth-work/Documents/code/KTIR-Scheduler/build && ninja scheduler
```

Expected: success, no warnings about new code.

- [ ] **Step 2: Full lit suite**

```bash
ninja check-scheduler 2>&1 | tail -15
```

Expected: 90 passed / 1 failed (pre-existing `pipeline-test.mlir`).

- [ ] **Step 3: Round-trip all the historical inputs**

```bash
for input in \
    /Users/prasanth-work/Documents/code/KTIR-Scheduler/test/Scheduler_pipeline/add_155_generic.mlir \
    /Users/prasanth-work/Documents/code/ktir-sdsc-translator/tests/sdsc/Softmax_225-Mul/ktir.mlir; do
  echo "=== $input ==="
  /Users/prasanth-work/Documents/code/KTIR-Scheduler/build/scheduler/bin/scheduler \
      --kEmitDFIR --mlir-print-ir-after=construct-three-stage-pipeline "$input" 2>&1 | \
      grep -A 5 "data_transfer" | head -20
done
```

Expected: each `data_transfer` carries inline affine subscripts. For the broadcast case, `[%arg0, %arg2, %arg3, 0]` should appear inline (constant `0` not a synthetic `arith.constant`).

- [ ] **Step 4: Confirm no leftover `arith.constant` zero-creation in createDataTransfers**

```bash
/Users/prasanth-work/Documents/code/KTIR-Scheduler/build/scheduler/bin/scheduler \
    --compute-group-extraction --construct-three-stage-pipeline \
    /Users/prasanth-work/Documents/code/ktir-sdsc-translator/tests/sdsc/Softmax_225-Mul/ktir.mlir 2>&1 | \
    grep -B 1 "data_transfer.*to %"
```

Expected: no more `%c0_NN = arith.constant 0` materialized just before each broadcast `data_transfer`. The `0` should appear inline in `[..., 0]`.

- [ ] **Step 5: No commit needed (verification only)**

If all four steps above pass, the implementation is done.

---

## Self-Review Notes (for the implementing agent)

If a step references something not defined elsewhere in this plan, treat it as a bug in the plan and pause to ask. Specifically watch for:

- **`getMixedSourceSizes` / `getMixedDestSizes` already existing**: Task 5 Step 3 says "if these methods already existed before Task 1's rewrite stripped them, just preserve the originals." Task 1's rewrite of the op definition keeps `let extraClassDeclaration = [{...}]` — verify whether the original block had these methods. If yes, both the declaration in Task 1 and the implementation in Task 5 must match the original signature.

- **Peek-ahead in the parser** (Task 3 Step 1): the `parser.getCurrentLocation().getPointer()[0] == '['` peek is a brittle workaround. The MLIR upstream `affine.dma_start` uses `parser.parseOptionalLSquare()` first; on success, the affine helper is then called with `Delimiter::None` (it expects no leading bracket). If your `parseAffineMapOfSSAIds` only supports `Delimiter::Square` (the default), the peek is required. If it accepts `Delimiter::None` cleanly, the upstream pattern is preferred. Inspect `mlir/include/mlir/IR/OpImplementation.h:1675-1679` and `mlir/lib/IR/AsmParser/Parser.cpp` to confirm.

- **Existing CHECK regressions**: Task 6 Step 5 anticipates that some `test/Transform/*.mlir` CHECK lines may fail because the post-pass IR no longer materializes per-dim arith ops. Regenerate via `mlir/utils/generate-test-checks.py` if that happens, the same way we did when migrating `ComputeGroupExtraction` tests.

- **`isSourceMemRef` / `isDestinationMemRef` already existing**: the original op had `isSourceMemRef` and `isSourceFifo` in `extraClassDeclaration`. The Task 1 rewrite preserves them and adds `isDestinationMemRef` / `isDestinationFifo`. Verify no duplicate decl.
