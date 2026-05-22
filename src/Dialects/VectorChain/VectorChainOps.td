//===-- VectorChain.td -- VectorChain Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2025
//
// =============================================================================
//
// This file defines the vectorchain operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTORCHAIN_TD
#define DIALECT_VECTORCHAIN_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"

#define COMMENT_OUT_COALESCE_STORE

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Dialects/Dataflow/DataflowIBMFPTypes.td"
include "src/Dialects/VectorChain/VectorChainEnums.td"

def VectorChain_Dialect : Dialect {
  let name = "vectorchain";
  let cppNamespace = "::mlir::vectorchain";

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "dataflow::DataflowDialect"
  ];
}

def VectorChainBinaryOperatorAttr : EnumAttr<VectorChain_Dialect, VectorChainBinaryOperator, "binary_operator">;
def VectorChainElementWiseCompareOperatorAttr : EnumAttr<VectorChain_Dialect, VectorChainElementWiseCompareOperator, "element_wise_compare_operator">;
def VectorChainExpEstimateAttr : EnumAttr<VectorChain_Dialect, VectorChainExpEstimate, "exp_estimate">;
def VectorChainEstimateVersionsAttr : EnumAttr<VectorChain_Dialect, VectorChainEstimateVersions, "estimate_versions">;
def VectorChainScanOpEvalOrdersAttr : EnumAttr<VectorChain_Dialect, VectorChainScanOpEvalOrders, "eval_order">;

// Base class for VectorChain dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class VectorChain_Op<string mnemonic, list<Trait> traits = []> :
    Op<VectorChain_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.

}

def VectorChain_SelectOp:VectorChain_Op<"select",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to shuffle/select required data";
    let description = [{
    "This operation selects/shuffle the required data."
    ```mlir
        %data = vectorchain.select %data {selection_map} : vector<4xi8>, vector<128xi8>
    ```
    }];
    let arguments = (ins AnyVectorOfAnyRank:$data, AffineMapAttr:$selection_map);
    let results = (outs AnyVectorOfAnyRank:$selected_data);
    let assemblyFormat = [{
         $data attr-dict `:` type($data) `,` type(results)
    }];
}

def VectorChain_MultiplyOp:VectorChain_Op<"multiply",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Element-wise multiply operation";
    let description = [{
    "This operation performs an element-wise multiply operations and reduce using the reduction map."
    ```mlir
        %data = vectorchain.multiply"(%op1, %op2) [%mask] {#reduction_map}
            : vector<128xi8>, vector<128xi8>, vector<64xi8>
    ```
    }];
    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName, AffineMapAttr:$reduction_map);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2 (`[` $mask^ `:` type($mask) `]`)?   attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_BinaryOp:VectorChain_Op<"binary",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Element-wise binary operation";
    let description = [{
    "This operation performs a binary element-wise operation (and, or, min, shuffle...) and reduce using the reduction map."
    ```mlir
        %data = vectorchain.binary"(%op1, %op2) [%mask] {#op_specific_map}
            : vector<128xi8>, vector<128xi8>, vector<128xi8>
    ```
    }];
    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         VectorChainBinaryOperatorAttr:$binary_op,
                         AffineMapAttr:$op_specific_map);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2  (`[` $mask^ `:` type($mask) `]`)?   attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_ElementWiseCompareOp:VectorChain_Op<"element_wise_compare",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Element-wise compare operation";
    let description = [{
    "This operation performs a element-wise compare operation"
    }];
    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         VectorChainElementWiseCompareOperatorAttr:$compare_op);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2  (`[` $mask^ `:` type($mask) `]`)?   attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_MergeOp:VectorChain_Op<"merge",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Merge operation";
    let description = [{
    "This operation performs a merge between two vectors based on the polyhedral mini-scop representation (where the order doesn't matter)."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2,
                         OptionalAttr<StrAttr>:$dbgName,
                         Builtin_IntegerSetAttr:$iteration_space_CA,
                         AffineMapAttr:$access_function_CA,
                         AffineMapAttr:$access_function_A,
                         Builtin_IntegerSetAttr:$iteration_space_CB,
                         AffineMapAttr:$access_function_CB,
                         AffineMapAttr:$access_function_B
                         );

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2    attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_CreateAffineMaskOp:VectorChain_Op<"create_affine_mask",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Create affine mask operation";
    let description = [{"Create affine mask operation"}];

    let arguments = (ins Optional<Index>:$mask_parameter,
                         Builtin_IntegerSetAttr:$mask_set);

    let results = (outs VectorOfRankAndType<[1], [I1]>:$data);
    let assemblyFormat = [{
        ($mask_parameter^ )? attr-dict `:` type(results)
    }];
}

def VectorChain_PackOp:VectorChain_Op<"pack",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Pack operation";
    let description = [{
    "This operation packs two vectors into a one based on the array of indices."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         ArrayAttr:$indices,
                         IndexAttr:$repetition,
                         DefaultValuedAttr<BoolAttr, "false">:$sign_extend);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2 (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_CastOp:VectorChain_Op<"cast",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Cast Operation";
    let description = [{
    "This operation casts a vector of one type to a vector of another."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$input, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $input (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($input) `,` type(results)
    }];
    let hasVerifier = 1;
}

def VectorChain_FastExpOp:VectorChain_Op<"fast_exp",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Fast Exponent Operation";
    let description = [{
    "This operation computes fast exp auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}



def VectorChain_ExpEstimateOp:VectorChain_Op<"exp_estimate",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates exp auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName,
                         VectorChainExpEstimateAttr:$version);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_FloorOp:VectorChain_Op<"floor",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Floor operation";
    let description = [{
    "This operation computes the floor function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_RecEstimateOp:VectorChain_Op<"rec_estimate",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates rec auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_LnEstimateOp:VectorChain_Op<"ln_estimate",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates ln auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_RsqrtEstimateOp:VectorChain_Op<"rsqrt_estimate",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates rsqrt auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_SigmoidEstimateOp:VectorChain_Op<"sigmoid_estimate",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates sigmoid auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName,
                         VectorChainEstimateVersionsAttr:$version);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_TanhEstimateOp:VectorChain_Op<"tanh_estimate",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates tanh auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName,
                         VectorChainEstimateVersionsAttr:$version);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_NegOp:VectorChain_Op<"neg",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Negation operation";
    let description = [{
    "This operation negates the input SSA variable."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_MultiplyAndAccumulateOp:VectorChain_Op<"multiply_and_accumulate",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Multiply and accumulate operation";
    let description = [{
    "This operation performs a mac operations -- op1 x op2 + op3 with reduction map."
    ```mlir
    %data = vectorchain.multiply_and_accumulate(%op1, %op2, %op3) [%mask] {#reduction_map}
            : vector<128xi8>, vector<128xi8>, vector<256xi8>, vector<64xi8>
    ```
    }];
    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2,
                         AnyVectorOfAnyRank:$op3, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         AffineMapAttr:$reduction_map);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2 `,` $op3 (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op1) `,` type($op2) `,` type($op3) `,` type(results)
    }];
}

def VectorChain_ElementWiseSelectionOp:VectorChain_Op<"element_wise_selection",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Element wise selection";
    let description = [{
    "This operation performs a element-wise selection b/w lhs and rhs operands based on the condition."
    }];
    let arguments = (ins AnyVectorOfAnyRank:$cond, AnyVectorOfAnyRank:$lhs,
                         AnyVectorOfAnyRank:$rhs, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $cond `?` $lhs `:` $rhs (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($cond) `,` type($lhs) `,` type($rhs) `,` type(results)
    }];
}

// PC: Removed ConstantLike trait of the ConstantBitstreamOp because MLIR canonicalization
// passes require ConstantLikeOp's to have foldable implementation.
// error: "expected ConstantLike op to be foldable"
def VectorChain_ConstantBitstreamOp:VectorChain_Op<"constant_bitstream",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Constant bitstream operation";
    let description = [{
    "This operation interprets a constant bit stream as a value with the given vector type.

    The value is interpretted as a signed 64 bit integer.
    
    The number of elements of the value attribute should match the number of elements in the
    result vector.

    The bitwidth of values specified by the value attribute should not exceed the bitwidth
    of the output element type."
    }];

    let arguments = (ins ArrayAttr:$value);

    let results = (outs AnyTypeOf<[Dataflow_IBMVectorFloatType, AnyVectorOfAnyRank]>:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def VectorChain_ShuffleOp:VectorChain_Op<"shuffle",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Shuffle operation";
    let description = [{
    "This operation shuffles the elements of a vector based on the array of indices.

    The elements of the indices refer to the elements of the input vector. The input should
    be vector type.

    The product of the number of elements of indices and the repetitions must match the
    number of elements in the output.

    A value of -1 indicates padding."
    }];

    let arguments = (ins AnyTypeOf<[Dataflow_IBMVectorFloatType, AnyVectorOfAnyRank]>:$input,
                         Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         ArrayAttr:$indices,
                         I32Attr:$repetition);

    let results = (outs AnyVectorOfAnyRank:$output);
    let assemblyFormat = [{
          $input (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($input) `,` type(results)
    }];

    let hasVerifier = 1;
}

def VectorChain_ScanWithGapOp:VectorChain_Op<"scan_with_gap",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Scan (reduction) operation with gap";
    let description = [{
    "This operation performs reduction using the given binary operation
    in reduction_op attribute. The gap attribute shows how many element
    one needs to jump to get to the next element. The eval_order is giving
    of reduction.
    Mathematical definition of the operation on n elements for eval_order=left-to-right:
    O[i] = sum {I[i - k * gap] | k = 0, ..., floor_div(i, gap)} for i=0,1,...,n
    "
    }];

    let arguments = (ins AnyVectorOfAnyRank:$input,
                         OptionalAttr<StrAttr>:$dbgName,
                         VectorChainBinaryOperatorAttr:$reduction_op,
                         IndexAttr:$gap,
                         VectorChainScanOpEvalOrdersAttr:$eval_order);

    let results = (outs AnyVectorOfAnyRank:$output);
    let assemblyFormat = [{
          $input attr-dict `:` type($input) `,` type(results)
    }];
}

def VectorChain_RotateOp:VectorChain_Op<"rotate",
                                           [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Rotation operation";
    let description = [{
        "This operation rotates the elements of a vector from a given location either left
        or right depending on the right_shift attribute. The operation views the vector
        as a circular array. If its a mult-dimensional array, then this operation views
        the vector in a linearized form and performs the operation."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$input,
                       Index:$position,
                       OptionalAttr<StrAttr>:$dbgName,
                       DefaultValuedAttr<BoolAttr, "true">:$right_shift);

    let results = (outs AnyVectorOfAnyRank:$output);
    let assemblyFormat = [{
        $input `,` $position attr-dict `:` type($input) `,` type($position) `,` type(results)
    }];
}

#ifdef COMMENT_OUT_COALESCE_STORE
#else
def VectorChain_CoalesceOp:VectorChain_Op<"coalesce",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to mask input data for coalesce store";
    let description = [{
    "This operation coalesces the required data with desired masking pattern."
    ```mlir
        %data = vectorchain.coalesce %data {drop_first, subword_len, stride} : vector<128xi8>, vector<64xi8>
    ```
    }];
    let arguments = (ins AnyVectorOfAnyRank:$data,
                        Variadic<Index>:$drop_first_indices,
                        OptionalAttr<StrAttr>:$dbgName,
                        AffineMapAttr:$drop_first_map,
                        I32Attr:$subword_len,
                        I32Attr:$stride);
    let results = (outs AnyVectorOfAnyRank:$coalesced_data);
    let verifier = [{ return ::verify(*this); }];
    let assemblyFormat = [{
         $data `drop_first_indices` `:` `[` $drop_first_indices `]` attr-dict `:` type($data) `,` type(results)
    }];
}
#endif

#endif // DIALECT_VECTORCHAIN_TD
