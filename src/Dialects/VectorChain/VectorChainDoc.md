# 'vectorchain' Dialect

## Throughout this write-up:

  _dbgName_ is an optional string parameter which, if used, refers back to a section of the parent 
          representation (e.g. another IR, json).

  _mask_ is an optional vector of boolean. Where mask is set to 1, those locations are not changed.

## Operations

### `vectorchain.binary` (vectorchain::BinaryOp)

_Element-wise operations on two inputs (so "binary" in this sense). Some (and, or, min) are bitwise._

Syntax:

```
operation ::= `vectorchain.binary` $op1 `,` $op2  (`[` $mask^ `:` type($mask) `]`)?  (`[` $dbgName^ `:` type($dbgName) `]`)? {#binary_op, #op_specific_map}  attr-dict `:` type($op1) `,` type($op2) `,` type(results)
```

"This operation performs an element-wise operation (e.g. and, or, min) and reduce using the reduction map. The map is usually the identity. Near the end of this document, there is a list of _built-in operators supported by VectorChain binary ops_."
```
Example:
    %res = vectorchain.binary %nfwd0, %lrf0_2 {binary_op = #vectorchain<binary_operator min>, 
           op_specific_map = affine_map<(i)->(i)>} : vector<64xf16>, vector<64xf16>, vector<64xf16>

This will find the minimum (pair-wise) of corresponding elements in f16 arrays %nfwd0 and %lrf0_2 and store
the results in f16 array %res. 
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>binary_op</code></td><td>::mlir::vectorchain::VectorChainBinaryOperatorAttr</td><td>built-in operators supported by VectorChain binary ops.</td></tr>
<tr><td><code>op_specific_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op1` | vector of any type values
| `op2` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.cast` (vectorchain::CastOp)

_Cast Operation_


Syntax:

```
operation ::= `vectorchain.cast` $input (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)?  attr-dict `:` type($input) `,` type(results)
```

"This operation casts a vector of one type to a vector of another. 
 (See also deeptools/util/sen_data_convert.cpp for supported type conversions)"
```
Example:

    %fp16_out = vectorchain.cast %fp32_inp [%mask] [%dbgName] : vector<8xf32>, vector<8xf16>

This will convert the fp32 input vector to fp16 and store in the output vector.
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.constant_bitstream` (vectorchain::ConstantBitstreamOp)

_Constant bitstream operation_

Syntax:

```
operation ::= `vectorchain.constant_bitstream` {value = [.., ..]}  attr-dict `:` type(results)
```

"This operation interprets a constant bit stream as a value with the given vector type."

- Each element of value is interpreted as a bit stream used to set the corresponding element of the result vector. Note that the elements of value are stored internally as signed 64-bit ints (although they are interpreted as bit streams).
- The number of elements of the value attribute should match the number of elements in the result vector.
- The bitwidth of values specified by the value attribute should not exceed the bitwidth of the output element type.

Examples:

   %n2 = vectorchain.constant_bitstream {value = [0xff, 0x0]} : vector<2xi8>

   %m4 = vectorchain.constant_bitstream {value = [0xff, 0xfe, 0xfd, 0xfc]} : vector<4xf16>

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::ArrayAttr</td><td>array attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` |  or vector of any type values


### `vectorchain.create_affine_mask` (vectorchain::CreateAffineMaskOp)

_Create affine mask operation_


Syntax:

```
operation ::= `vectorchain.create_affine_mask` (`[` $mask^ `:` type($mask) `]`)? {#mask_set }  attr-dict `:` type(results)
```

"Create affine mask operation from the input mask_set. Mask_set captures indices of vector that needs to be masked."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mask_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `mask_parameter` | index

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.element_wise_compare` (vectorchain::ElementWiseCompareOp)

_Element-wise compare operation_


Syntax:

```
operation ::= `vectorchain.element_wise_compare` $op1 `,` $op2  (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)? {#compare_op}  attr-dict `:` type($op1) `,` type($op2) `,` type(results)
```

"This operation performs an element-wise compare operation. Near the end of this file is list of _built-in operators supported by VectorChain Compare ops_"

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>compare_op</code></td><td>::mlir::vectorchain::VectorChainElementWiseCompareOperatorAttr</td><td>built-in operators supported by VectorChain Compare ops.</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op1` | vector of any type values
| `op2` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.element_wise_selection` (vectorchain::ElementWiseSelectionOp)

_Element wise selection_


Syntax:

```
operation ::= `vectorchain.element_wise_selection` $cond `?` $lhs `:` $rhs (`[` $mask^ `:` type($mask) `]`)?  (`[` $dbgName^ `:` type($dbgName) `]`)?  attr-dict `:` type($cond) `,` type($lhs) `,` type($rhs) `,` type(results)
```

"This operation performs a element-wise selection between lhs and rhs operands based on the condition.
(Similar to 'cond ? lhs : rhs' scalar operation in C)"

```
Example:
%result = vectorchain.element_wise_selection %cond ? %lhs : %rhs : vector<4xi1>, vector<4xf32>, vector<4xf32>, vector<4xf32>

Pseudo-code:
    // loop over cond vector elements:
    for (i = 0, elements_in_cond - 1){
       if ( mask == NULL || mask[i] == 0 ){
          if ( cond[i] ){
            result[i] = lhs[i]
          } else {
            result[i] = rhs[i]
          }
       }
    }
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `cond` | vector of any type values
| `lhs` | vector of any type values
| `rhs` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.exp_estimate` (vectorchain::ExpEstimateOp)

_Estimate operation_

Syntax:

```
operation ::= `vectorchain.exp_estimate` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)? {#version}  attr-dict `:` type($op) `,` type(results)
```

"This operation estimates the exp auxiliary function (exponential), based on #version (a or b),
 of the values in the input array and stores them in the output array."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>version</code></td><td>::mlir::vectorchain::VectorChainExpEstimateAttr</td><td>List of versions of exp func extimates.</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.fast_exp` (vectorchain::FastExpOp)

_Fast Exponent Operation_


Syntax:

```
operation ::= `vectorchain.fast_exp` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)?  attr-dict `:` type($op) `,` type(results)
```

"This operation computes fast exp auxiliary function (exponential)
 of the values in the input array and stores them in the output array."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.ln_estimate` (vectorchain::LnEstimateOp)

_Estimate operation_


Syntax:

```
operation ::= `vectorchain.ln_estimate` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)?  attr-dict `:` type($op) `,` type(results)
```

"This operation estimates ln auxiliary function (natural logarithm)
 of the values in the input array and stores them in the output array."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.multiply` (vectorchain::MultiplyOp)

_Element-wise multiply operation_


Syntax:

```
operation ::= `vectorchain.multiply` $op1 `,` $op2 (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)? {#reduction_map}  attr-dict `:` type($op1) `,` type($op2) `,` type(results)
```

"This operation performs element-wise multiply operations and reduce using the reduction map: result += op1\*op2.
 1. element-wise multiplication of vectors op1 and op2
 2. reduce multiplied results as specified by “reduction_map” \
 Note: the mask and results vectors use the same index."
```mlir
    %results = vectorchain.multiply"(%op1, %op2) [%mask] [%dbgName] {#reduction_map}
        : vector<128xi8>, vector<128xi8>, vector<64xi8>

Pseudo-code:

    // number of elements in op1 and op2 should match
    if (elements_in_op1 != elements_in_op2 ) error_exit
    for (i = 0, (elements_in_op1 - 1) ){
       im = reduction_map(i)
       if (mask == NULL || mask[im] == 0 ){
          // in example all are i8 - in general, cast could be needed here
          results[im] += op1[i]*op2[i]
       }
    }
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>reduction_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op1` | vector of any type values
| `op2` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.multiply_and_accumulate` (vectorchain::MultiplyAndAccumulateOp)

_Multiply and accumulate operation_


Syntax:

```
operation ::= `vectorchain.multiply_and_accumulate` $op1 `,` $op2 `,` $op3 (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)? {#reduction_map} attr-dict `:` type($op1) `,` type($op2) `,` type($op3) `,` type(results)
```

"This operation performs a mac (multiply-accumulate) operation -- op1 x op2 + op3 with reduction map.
 
 1. element-wise multiplication of vectors op1 and op2
 2. reduce (accumulate) multiplied results as specified by “reduction_map”
 3. element-wise addition of results with vector op3 \
 Note: the mask and results vectors use the same index." 
```mlir
%results = vectorchain.multiply_and_accumulate(%op1, %op2, %op3) [%mask] [%dbgName] {#reduction_map}
        : vector<128xi8>, vector<128xi8>, vector<256xi8>, vector<64xi8>

Pseudo-code:

    // number of elements in op1 and op2 should match
    if (elements_in_op1 != elements_in_op2 ) error_exit
    for (i = 0, (elements_in_op1 - 1) ){
       im = reduction_map(i)
       if (mask == NULL || mask[im] == 0 ){
          // in example all are i8 - in general, cast could be needed here
          results[im] += op1[i]*op2[i] + op3[im]
       }
    }
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>reduction_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op1` | vector of any type values
| `op2` | vector of any type values
| `op3` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.neg` (vectorchain::NegOp)

_Negation operation_


Syntax:

```
operation ::= `vectorchain.neg` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)?  attr-dict `:` type($op) `,` type(results)
```

"This operation negates the values in the input SSA variable (array) and store the results
 in the output array."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.pack` (vectorchain::PackOp)

_Pack operation_


Syntax:

```
operation ::= `vectorchain.pack` $op1 `,` $op2 (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)?{$indices, $repetition} (`[` $sign_extend^ `:` type($sign_extend) `]`)?  attr-dict `:` type($op1) `,` type($op2) `,` type(results)
```

"This operation packs two vectors into one based on the array of indices."
 - The input arrays must have the same number of elements.
 - Input and output may have different data-types.
 - The number of elements in the indices vector must equal number_of_elements_in_result divided by repetition.
 - Value of -1 in indices means the result in that location is 0?
 - The default value of sign_extend is false.

For integer data:
- sign_extend=false means input values are padded with zeroes, if needed. (e.g. 4-bit "0xd" goes to 8-bit "0x0d")
- sign_extend=true  means sign bit of non-zero input value will be used for padding. (e.g. 4-bit "0xd" goes to 8-bit "0xfd")
An input value of exactly zero will be padded with zeroes, if needed.

```
General case:

%output = vectorchain.pack %op1, %op2 ([%mask: type(%mask)])? {$indices, $repetition} ([%sign_extend: type($sign_extend)])  
          : type($op1), type($op2), type(output)

Example:

%res = vectorchain.pack %X, %Z {indices = [13, 14, 15, 3, 4, 5, 6, 7], repetition=8:index, sign_extend=false} :
        vector<64xf16>, vector<64xf16>, vector<64xf16

Input locations: (where the indices length of 8 must match 64/8 with 64 from "vector<64xf16>", 8 from repetition=8)
X:  [0, 1,  2,  3,  4,  5,  6,  7]
Z:  [8, 9, 10, 11, 12, 13, 14, 15]  Note: real locations of Z are 0-7
indices = [13, 14, 15, 3, 4, 5, 6, 7]

There are 8 slices, one for each repetition.

First slice: 
result[0 - 7] = [  Z[5],  Z[6],  Z[7],  X[3],  X[4],  X[5],  X[6],  X[7] ]

Second slice: add 1*indices_length to first slice locations
result[8 -15] = [ Z[13], Z[14], Z[15], X[11], X[12], X[13], X[14], X[15] ]

etc.

Eighth(repetition-th) slice: add (repetition-1)*indices_length to first slice locations
result[56-63] = [ Z[61], Z[62], Z[63], X[59], X[60], X[61], X[62], X[63] ]


Another Example: (number of input and output elements differ)

    %11 = vectorchain.pack %9, %10 {indices = [0, 1, 2, 3, 4, 5, 6, 7], repetition = 8 : index} 
          : vector<32xf16>, vector<32xf16>, vector<64xf16>

Pseudo-code:

    // number of elements in op1 and op2 should match
    if (elements_in_op1 != elements_in_op2 ) error_exit
    out_slice_size = elements_in_output / repetitions
    // elements_in_indices and output slice_size should match
    if ( elements_in_indices != out_slice_size ) error_exit
    // inp_slice_size and out_slice_size may differ.
    inp_slice_size = elements_in_op1 / repetitions
    do_sign_extend = false
    if (sign_extend ){
        if ( op1 and op2 are output are all integer types ){
           if (op1 and op2 and output are all signed ) {
              do_sign_extend = true
           } else if (op1 and op2 and output are all unsigned )  {
              do_sign_extend = true
           } else {
              // sign_extend true but types are not consistent
              error_exit
           }
        }
    }
    for irep = 0, (repetitions - 1) {
       for i_ind = 0, (elements_in_indices - 1){
           i_out = irep*out_slice_size + i_ind
           if (mask == NULL || mask[i_out] == 0) {
              i_inp = irep*inp_slice_size + indices[i_ind]
              if (indices[i_ind] >= inp_slice_size ) {
                 i_inp = i_inp - inp_slice_size
                 // tmp and output have the same type
                 if (do_sign_extend) {
                   tmp = op2[i_inp]  // sign extend if needed
                 } else {
                   tmp = op2[i_inp]  // cast if needed
                 }
                 output[i_out] = tmp 
              } else {
                 if (indices[i_ind] >= 0){
                   // tmp and output have same type
                   if (do_sign_extend) {
                     tmp = op1[i_inp]  // sign extend if needed
                   } else {
                     tmp = op1[i_inp]  // cast if needed
                   }
                   output[i_out] = tmp 
                 } else {
                   // or should this be skipped???
                   output[i_out] = zero // of the correct type
                 }
              }
           }
       }
    }
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>indices</code></td><td>::mlir::ArrayAttr</td><td>array attribute</td></tr>
<tr><td><code>repetition</code></td><td>::mlir::IntegerAttr</td><td>index attribute</td></tr>
<tr><td><code>sign_extend</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op1` | vector of any type values
| `op2` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.rec_estimate` (vectorchain::RecEstimateOp)

_Estimate operation_


Syntax:

```
operation ::= `vectorchain.rec_estimate` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)?  attr-dict `:` type($op) `,` type(results)
```

"This operation estimates the reciprocal auxiliary function of the values in the
input array and stores them in the output array. (result = 1/op)"

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.rotate` (vectorchain::RotateOp)

_Rotation operation_


Syntax:

```
operation ::= `vectorchain.rotate` $input `,` $position (`[` $dbgName^ `:` type($dbgName) `]`)? (`[` $right_shift^ `:` type($right_shift) `]`)?  attr-dict `:` type($input) `,` type($position) `,` type(results)
```

"This operation rotates the elements of a vector from a given location either left
or right depending on the right_shift attribute. The operation views the vector
as a circular array. If its a multi-dimensional array, then this operation views
the vector in a linearized form and performs the operation. $position is the number
of elements by which to rotate the data. The $right_shift boolean defaults to true if omitted."

```
Example:

input:
A B C D (vector size = 4)

position=0 + right-shift --> A B C D
position=1 + right-shift --> D A B C
position=2 + right-shift --> C D A B
position=3 + right-shift --> B C D A

position=0 + left-shift --> A B C D
position=1 + left-shift --> B C D A
position=2 + left-shift --> C D A B
position=3 + left-shift --> D A B C

```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>right_shift</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | vector of any type values
| `position` | index

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | vector of any type values


### `vectorchain.rsqrt_estimate` (vectorchain::RsqrtEstimateOp)

_Estimate operation_


Syntax:

```
operation ::= `vectorchain.rsqrt_estimate` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)?  attr-dict `:` type($op) `,` type(results)
```

"This operation estimates rsqrt auxiliary function (reciprocal square root) of the values in the
input array and stores them in the output array."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.scan_with_gap` (vectorchain::ScanWithGapOp)

_Scan (reduction) operation with gap_


Syntax:

```
operation ::= `vectorchain.scan_with_gap` $input (`[` $dbgName^ `:` type($dbgName) `]`)? {#reduction_op, $gap, $eval_order}  attr-dict `:` type($input) `,` type(results)
```

"This operation performs reduction using the given binary operation
in the reduction_op attribute. The gap attribute shows how many elements
one needs to jump to get to the next element. The eval_order indicates the
order in which the inputs are reduced to produce an output element."

```
Mathematical definition of the reduction operation (sum) on n elements for eval_order=left-to-right:
O[i] = sum {I[i - k * gap] | k = 0, ..., floor_div(i, gap)} for i=0,1,...,n
(note: O[m * gap+j] = sum {I[m * gap+j], O[(m-1) * gap+j]}

So for left to right:
i=0: k=0
O[0] = I[0]
i=1: k=0
O[1] = I[1]
...
i=(gap -1): k=0
O[gap-1] = I[gap -1]

i=gap: k=0,1
O[gap] = sum {I[gap], I[0]}
i=gap+1: k=0,1
O[gap+1] = sum {I[gap+1], I[1]}
...
i=(2 * gap-1): k=0,1
O[2 * gap-1] = sum {I[2 * gap -1], I[gap -1]}

i=2 * gap: k=0,1,2
O[2 * gap] = sum {I[2 * gap], I[gap], I[0]}
i=2 * gap+1: k=0,1,2
O[gap+1] = sum {I[2 * gap+1], I[gap+1], I[1]}
...
i=(3 * gap -1): k=0,1,2
O[3 * gap-1] = sum {I[3 * gap -1], I[2 * gap -1], I[gap -1]}


Is this needed??? If so, is the following correct??
Mathematical definition of the reduction operation (sum) on n elements for eval_order=right-to-left:
O[n-i] = sum {I[n-i -k * gap] | k = 0, ..., floor_div(i, gap)} for i=0,1,...,n
(note: O[n-m * gap-j] = sum {I[n-m * gap-j], O[n-(m-1) * gap-j]})

so for right-to-left:
i=0: k=0
O[n] = I[n]
i=1: k=0
O[n-1] = I[n-1]
i=gap-1: k=0
O[n-gap+1] = I[n-gap+1]

i=gap: k=0,1
O[n-gap] = sum {I[n], I[n-gap]}
i=gap+1: k=0,1
O[n-gap-1] = sum {I[n-1], I[n-gap-1]}
...
i=(2 * gap-1): k=0,1
O[n-2 * gap+1] = sum {I[n-gap+1], I[n-2 * gap+1]}

i=2 * gap: k=0,1,2
O[n-2 * gap] = sum {I[n], I[n-gap], I[n-2 * gap]}
i=2 * gap+1: k=0,1,2
O[n-2 * gap-1] = sum {I[n-1], I[n-gap-1], I[n-2 * gap-1]}
...
i=(3 * gap -1): k=0,1,2
O[n-3 * gap+1] = sum {I[n-gap+1], I[n-2 * gap+1], I[n-3 * gap+1]}
etc.
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>reduction_op</code></td><td>::mlir::vectorchain::VectorChainBinaryOperatorAttr</td><td>built-in operators supported by VectorChain binary ops.</td></tr>
<tr><td><code>gap</code></td><td>::mlir::IntegerAttr</td><td>index attribute</td></tr>
<tr><td><code>eval_order</code></td><td>::mlir::vectorchain::VectorChainScanOpEvalOrdersAttr</td><td>Reduction evaluation order.</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | vector of any type values


### `vectorchain.select` (vectorchain::SelectOp)

_Operation to shuffle/select required data_


Syntax:

```
operation ::= `vectorchain.select` $data {#selection_map}  attr-dict `:` type($data) `,` type(results)
```

"This operation selects/shuffle the required data based on the input map."
```mlir

Example:
      #map2 = affine_map<(d0) -> (d0 mod 16)>
      %results = vectorchain.select %input {selection_map = #map2} : vector<16xi8>, vector<256xi8>

Interpreted as:
      results[i] = input[map(i)]"
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>selection_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `selected_data` | vector of any type values


### `vectorchain.shuffle` (vectorchain::ShuffleOp)

_Shuffle operation_


Syntax:

```
operation ::= `vectorchain.shuffle` $input (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)? {$indices, $repetition}  attr-dict `:` type($input) `,` type(results)
```

"This operation shuffles the elements of a vector based on the array of indices."

 - The elements of the indices refer to the elements of the input vector. The input should be vector type.
 - The product of the number of elements of indices and the repetitions must match the number of elements in the output.
 - A value of -1 indicates padding.

```
Examples:

%38 = vectorchain.shuffle %37 {indices = [8, 9, 10, 11, 12, 13, 14, 15], repetition = 8 : i32} : vector<128xf16>, vector<64xf16>

%splat = vectorchain.shuffle %inp {indices = [0], repetition =64 : i32} : vector<1x1x1x1xf16>, vector<1x1x1x64xf16

Psuedo-code:
   // input slice size
   inp_slice_len = elements_in_input/repetitions
   // output slice size - should match elements_in_indices
   out_slice_len = elements_in_output/repetitions
   if (elements_in_indices != out_slice_len ) exit_with_error
   for irep = 0 to (repetitions - 1){
     for i_ind = 0, (elements_in_indices - 1){
        i_out = irep * out_slice_len + i_ind
        if ( mask == NULL || mask[i_out] == 0 ){
          if (indices[i_ind] >= 0){
            i_inp = irep * inp_slice_len + indices[i_ind]
            output[i_out] = input[i_inp]
          }
        }
     }
   }
```

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>indices</code></td><td>::mlir::ArrayAttr</td><td>array attribute</td></tr>
<tr><td><code>repetition</code></td><td>::mlir::IntegerAttr</td><td>32-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` |  or vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | vector of any type values


### `vectorchain.sigmoid_estimate` (vectorchain::SigmoidEstimateOp)

_Estimate operation_


Syntax:

```
operation ::= `vectorchain.sigmoid_estimate` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)? {#version}  attr-dict `:` type($op) `,` type(results)
```

"This operation estimates sigmoid auxiliary function, based on #version (slope or offset),
 of the values in the input array and stores the results in the output array."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>version</code></td><td>::mlir::vectorchain::VectorChainEstimateVersionsAttr</td><td>Slope or offset of the func extimates.</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


### `vectorchain.tanh_estimate` (vectorchain::TanhEstimateOp)

_Estimate operation_


Syntax:

```
operation ::= `vectorchain.tanh_estimate` $op (`[` $mask^ `:` type($mask) `]`)? (`[` $dbgName^ `:` type($dbgName) `]`)? {#version}  attr-dict `:` type($op) `,` type(results)
```

"This operation estimates tanh (hyperbolic tangent) auxiliary function, based on #version (slope or offset),
of the values in the input array and stores them in the output array."

Traits: `IsolatedFromAbove`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>version</code></td><td>::mlir::vectorchain::VectorChainEstimateVersionsAttr</td><td>Slope or offset of the func extimates.</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | vector of any type values
| `mask` | vector of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | vector of any type values


## Attributes

### VectorChainBinaryOperatorAttr

built-in operators supported by VectorChain binary ops.

Syntax:

```
#vectorchain.binary_operator<
  ::mlir::vectorchain::VectorChainBinaryOperator   # value
>
```

Enum cases:
* and0 (`and0`)
* or0 (`or0`)
* xnor (`xnor`)
* and_not (`and_not`)
* min (`min`)
* max (`max`)
* abs_min (`abs_min`)
* abs_max (`abs_max`)
* add (`add`)
* mul (`mul`)
* sub (`sub`)
* mul_div2 (`mul_div2`)

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::vectorchain::VectorChainBinaryOperator` | an enum of type VectorChainBinaryOperator |

### VectorChainElementWiseCompareOperatorAttr

built-in operators supported by VectorChain Compare ops.

Syntax:

```
#vectorchain.element_wise_compare_operator<
  ::mlir::vectorchain::VectorChainElementWiseCompareOperator   # value
>
```

Enum cases:
* compare_eq (`compare_eq`)
* compare_neq (`compare_neq`)
* compare_lt (`compare_lt`)
* compare_le (`compare_le`)
* compare_gt (`compare_gt`)
* compare_ge (`compare_ge`)

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::vectorchain::VectorChainElementWiseCompareOperator` | an enum of type VectorChainElementWiseCompareOperator |

### VectorChainEstimateVersionsAttr

Slope or offset of the func extimates.

Syntax:

```
#vectorchain.estimate_versions<
  ::mlir::vectorchain::VectorChainEstimateVersions   # value
>
```

Enum cases:
* slope (`slope`)
* offset (`offset`)

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::vectorchain::VectorChainEstimateVersions` | an enum of type VectorChainEstimateVersions |

### VectorChainExpEstimateAttr

List of versions of exp func extimates.

Syntax:

```
#vectorchain.exp_estimate<
  ::mlir::vectorchain::VectorChainExpEstimate   # value
>
```

Enum cases:
* a (`a`)
* b (`b`)

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::vectorchain::VectorChainExpEstimate` | an enum of type VectorChainExpEstimate |

### VectorChainScanOpEvalOrdersAttr

Reduction evaluation order.

Syntax:

```
#vectorchain.eval_order<
  ::mlir::vectorchain::VectorChainScanOpEvalOrders   # value
>
```

Enum cases:
* left_to_right (`left_to_right`)
* right_to_left (`right_to_left`)

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::vectorchain::VectorChainScanOpEvalOrders` | an enum of type VectorChainScanOpEvalOrders |

## Enums

### VectorChainBinaryOperator

built-in operators supported by VectorChain binary ops.

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| and0 | `0` | and0 |
| or0 | `1` | or0 |
| xnor | `2` | xnor |
| and_not | `3` | and_not |
| min | `4` | min |
| max | `5` | max |
| abs_min | `6` | abs_min |
| abs_max | `7` | abs_max |
| add | `8` | add |
| mul | `9` | mul |
| sub | `10` | sub |
| mul_div2 | `11` | mul_div2 |

### VectorChainElementWiseCompareOperator

built-in operators supported by VectorChain Compare ops.

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| compare_eq | `0` | compare_eq |
| compare_neq | `1` | compare_neq |
| compare_lt | `2` | compare_lt |
| compare_le | `3` | compare_le |
| compare_gt | `4` | compare_gt |
| compare_ge | `5` | compare_ge |

### VectorChainEstimateVersions

Slope or offset of the func extimates.

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| slope | `0` | slope |
| offset | `1` | offset |

### VectorChainExpEstimate

List of versions of exp func extimates.

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| a | `0` | a |
| b | `1` | b |

### VectorChainScanOpEvalOrders

Reduction evaluation order.

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| left_to_right | `0` | left_to_right |
| right_to_left | `1` | right_to_left |

