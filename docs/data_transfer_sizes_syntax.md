# Data Transfer Sizes: Static and Dynamic Support

## Overview

The `ktdf.data_transfer` operation now supports both **constant (static) sizes** and **SSA operand (dynamic) sizes**, providing flexibility for different use cases while maintaining backward compatibility.

## Motivation

The original implementation used only SSA operands for sizes (e.g., `%c64`), which required defining constants via `arith.constant` operations. However, documentation examples showed cleaner syntax with inline constant values (e.g., `size [1, 64]`). This enhancement allows both approaches.

## Syntax

### Constant Sizes (New - Matches Documentation)

When sizes are known at compile time, you can specify them directly as integer literals:

```mlir
ktdf.data_transfer from %A[%m, %n] size [1, 64] to %lx_A[%c0] size [64]
  : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
```

**Benefits:**
- Cleaner, more readable IR
- Matches documentation examples
- Enables better optimization opportunities
- No need for separate `arith.constant` operations

### Dynamic Sizes (Existing - Still Supported)

When sizes are computed at runtime or depend on function arguments, use SSA values:

```mlir
%c1 = arith.constant 1 : index
%c64 = arith.constant 64 : index
ktdf.data_transfer from %A[%m, %n] size [%c1, %c64] to %lx_A[%c0] size [%c64]
  : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
```

**Benefits:**
- Full flexibility for runtime-computed sizes
- Supports parametric code
- Backward compatible with existing code

### Mixed Sizes (Best of Both Worlds)

You can mix constant and dynamic sizes in the same operation:

```mlir
func.func @example(%dynamic_size: index) {
  %c0 = arith.constant 0 : index
  %A = memref.alloc() : memref<1x64xf16, "DRAM">
  %lx_A = memref.alloc() : memref<64xf16, "LX">
  
  // First dimension is constant (1), second is dynamic
  ktdf.data_transfer from %A[%c0, %c0] size [1, %dynamic_size] 
                     to %lx_A[%c0] size [%dynamic_size]
    : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
}
```

## Implementation Details

### TableGen Definition

The operation now includes optional static size attributes:

```tablegen
let arguments = (ins
  AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$source,
  Variadic<Index>:$source_indices,
  Variadic<Index>:$source_sizes,
  AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$destination,
  Variadic<Index>:$dest_indices,
  Variadic<Index>:$dest_sizes,
  OptionalAttr<DenseI64ArrayAttr>:$static_source_sizes,
  OptionalAttr<DenseI64ArrayAttr>:$static_dest_sizes
);
```

### Custom Parser/Printer

A custom `parseSizesList` and `printSizesList` function handles the mixed syntax:

- **Parser**: Attempts to parse each element as an SSA value first, then falls back to integer literal
- **Printer**: Outputs constants directly, SSA values with their names
- **Encoding**: Static sizes use `DenseI64ArrayAttr`, dynamic positions marked with `ShapedType::kDynamic`

### Helper Methods

#### Mixed Size Getters (Static + Dynamic)

Two convenience methods retrieve sizes in a unified format:

```cpp
SmallVector<OpFoldResult> getMixedSourceSizes();
SmallVector<OpFoldResult> getMixedDestSizes();
```

These return `OpFoldResult` which can hold either:
- `IntegerAttr` for static sizes
- `Value` for dynamic sizes

#### Static-Only Size Getters

When you need to work with fully static sizes:

```cpp
// Check if all sizes are static (no dynamic values)
bool hasAllStaticSourceSizes();
bool hasAllStaticDestSizes();

// Get static sizes as integer array (returns std::nullopt if any size is dynamic)
std::optional<SmallVector<int64_t>> getStaticSourceSizesArray();
std::optional<SmallVector<int64_t>> getStaticDestSizesArray();
```

**Usage Example:**
```cpp
// Check if we can optimize based on static sizes
if (transferOp.hasAllStaticSourceSizes()) {
  auto sizes = transferOp.getStaticSourceSizesArray();
  // sizes is guaranteed to be present
  int64_t totalSize = 1;
  for (int64_t size : *sizes) {
    totalSize *= size;
  }
  // Perform compile-time optimization based on totalSize
}
```

## Migration Guide

### Existing Code (No Changes Required)

All existing code using SSA operands continues to work:

```mlir
// This still works exactly as before
%c64 = arith.constant 64 : index
ktdf.data_transfer from %src[%c0] size [%c64] to %dst[%c0] size [%c64]
  : memref<64xf16, "LX">, memref<64xf16, "SFP_REG">
```

### New Code (Recommended Style)

For known constant sizes, use the cleaner syntax:

```mlir
// Preferred for constants
ktdf.data_transfer from %src[%c0] size [64] to %dst[%c0] size [64]
  : memref<64xf16, "LX">, memref<64xf16, "SFP_REG">
```

### Optimization Opportunities

Passes can now:
1. Fold constant sizes for better analysis
2. Detect transfer patterns more easily
3. Perform size-based optimizations without tracing through `arith.constant` ops

## Examples from Documentation

All documentation examples now parse correctly:

```mlir
// From MatrixAdd.md - Path Expansion examples
ktdf.data_transfer from %A[%m, %n] size [1, 64] to %lx_A[%c0] size [64]
  : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">

ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot_a size [64]
  : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>

ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64]
  : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
```

## Testing

See `test/Dialects/KTDF/data-transfer-mixed-sizes.mlir` for comprehensive test cases covering:
- Pure constant sizes
- Pure dynamic sizes  
- Mixed constant and dynamic sizes
- FIFO transfers with both syntaxes

## Backward Compatibility

✅ **Fully backward compatible** - All existing code continues to work without modification.

## Future Work

- Consider adding canonicalization patterns to convert `arith.constant` + dynamic size → static size
- Add folder to simplify operations with all-static sizes
- Extend similar support to other operations that use size operands