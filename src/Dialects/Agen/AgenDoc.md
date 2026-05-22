# 'agen' Dialect

Notes: optional arguments such as num_memref_indices (appearing in AgenOps.td under vector_load) and
have_input_vector (under composite_store) were omitted from this documentation, since no example usage was
found in active files under dcc/test. In the following descriptions, address computations are generally 
done in units of element offsets for the data type of the memref or vector used by the operation. 
(So element offsets must be multiplied by the size of the data type to get byte offsets.)

## Operations

### `agen.vector_load` (agen::VectorLoadOp)

_Loads a vector from memory using affine maps for address computation. Supports optional multicast information._

**Syntax**:
```mlir
%result = agen.vector_load %memref[%indices...] multicast_info = optional_multicast_group {
                           dbgName = "optional_debug_name",
                           load_set = affine_set<(dimensions) : (constraints)>,
                           load_order = affine_map<(dimensions) -> (result_order)>
                           } : memref_type, result_vector_type 
```

**Arguments**:
- `memref`: Memory reference to load from, produced by dataflow.get_logical_memory_view.
- `indices`: Variadic indices (element offsets) indicating starting location in memref, 
  after permuation via load_order and evaluation of layout_map.
  (shown as `[%indices...]` in syntax, also called "subscripts map")
- `multicast_info`: Optional multicast_group, produced by dataflow.create_multicast_group.
- `dbgName`: Optional debug name.
- `load_set`: Integer set defining lower and upper bounds for element offsets from memref starting
   location, as computed from layout_map(load_order(indices)) + offset_from_memory_view.
  (shown as `load_set = affine_set<...>` in syntax)
- `load_order`: Affine map defining order in which the indices input to the subscripts map and the
  offset memref are varied (which index changes fastest, which changes slowest, etc. ... 
  shown as `load_order = affine_map<...>` in syntax)

**Example**
```
%xrf_memory_unit = dataflow.get_local_unit %pt_0_unit {name="ptxrf"} : index
%xrf_memory = dataflow.get_logical_memory_view %xrf_memory_unit, %zero
                         {layout_map = affine_map<(i, j, k) -> (1024 * i + 512 * j + 256 * k) >}
                         : index, index, memref<?x?x256xi8>

%ker0 = agen.vector_load %xrf_memory[%i7, %i8, %zero]
                         {load_set = affine_set<(d0, d1, d2): (d2 >= 0, -d2 + 255 >= 0, d1 == 0, d0 == 0)>,
                         load_order = affine_map<(d0, d1, d2)-> (d0, d1, d2)>}
                         : memref<?x?x256xi8>, vector<256xi8>

Notes:
-----
   Apply load_order (a permutation) to subscript map (inside the [..] parameter for agen.vector_load) and 
   substitute result into layout_map (from get_logical_memory_view). Output will be an "element offset" 
   into input memref (source of load, %xrf_memory here).

pseudo-code
-----------
   i8 ker0[256]
   new_xrf = (i8 *) ( %xrf_memory + layout_map( load_order( %i7, %i8, %zero)) + %zero ) // 2nd "%zero" from get_logical_memory_view
   // loop over indices as specified by load_order
   // ranges of indices come from load_set
   dst_addr = 0
   for k in 0, 256-1
       ker0[dst_addr++] = new_xrf[k]

```

**Results**: Vector of any rank

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`, `AffineReadOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>load_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>load_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | memref of any type values
| `indices` | variadic of index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | vector of any type values


### `agen.vector_store` (agen::VectorStoreOp)

_Stores a vector to memory using affine maps for address computation._

**Syntax**:
```mlir
agen.vector_store %value, %memref[%indices...] {
  dbgName = "optional_debug_name",
  store_set = affine_set<(dimensions) : (constraints)>,
  store_order = affine_map<(dimensions) -> (results)>
} : vector_type, memref_type
```

**Arguments**:
- `value`: Vector specifying source for store (shown as `%value` in syntax)
- `memref`: Memory reference specifying target of store (shown as `%memref` in syntax)
- `indices`: Variadic indices (element offsets) indicating the starting location in memref, 
  after permutation via store_order and evaluation of layout_map.
  (shown as `[%indices...]` in syntax, also called "subscripts map")
- `dbgName`: Optional debug name (shown as `dbgName = "..."` in syntax)
- `store_set`: Integer set defining lower and upper bounds for element offsets from memref starting
  location, as computed from layout_map(store_order(indices)) + offset_from_memory_view. 
  (shown as `store_set = affine_set<...>` in syntax)
- `store_order`: Affine map defining order in which the indices input to the subscripts map and the
  offset memref are varied (which index changes fastest, which changes slowest, etc. ... 
  shown as `store_order = affine_map<...>` in syntax)

**Example**:
```
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d1 >= 0, -d1 + 63 >= 0, d0 == 0)>
%2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "ptrow0_0", type = "ptrow0"} : index 
%20 = dataflow.get_local_unit %2 {name="ptxrf"} : index
%c0 = arith.constant 0 : index
%tgt = dataflow.get_logical_memory_view %20, %c0 {layout_map = #map0} : index, index, memref<64x64xf16>
agen.vector_store inp_vec, %tgt[%c4 + %arg0, 0] {store_order = #map1, store_set = #set} : memref<64x64xf16>, vector<64xf16>

Notes:
-----
   Apply store_order (a permutation) to subscript map (inside the [..] parameter for agen.vector_store) and 
   substitute result into layout_map (from get_logical_memory_view). Output will be an "element offset" 
   into the memref (target of store, %tgt here).


pseudo-code
-----------
   new_tgt = (f16 *)( %tgt + layout_map(store_order(%c4 + %arg0, 0)) + %c0) // %c0 from get_logical_memory_view
   // loop over indices as specified by store_order
   // ranges of indices come from store_set
   src_addr = 0
   for j in 0, 63
       new_tgt[j] = inp_vec[src_addr++]
```

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`, `AffineWriteOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>store_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>store_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `value` | any type
| `memref` | memref of any type values
| `indices` | variadic of index


### `agen.composite_load` (agen::CompositeLoadOp)

_Agen composite load operation_

**Description**: agen.composite_load is equivalent to agen.vector_load invoked from within a loop nest, where the bounds and ordering of the loops are defined by time_set and time_order. An additional element offset from the base memref (coded immediately after "agen.composite_load") is specified via time_addr_map.

**Syntax**:
```mlir
agen.composite_load %memref[%indices...] 
  time_symbols(optional_time_symbols) (iter_data : type) {
  dbgName = "optional_debug_name",
  load_set = affine_set<(dimensions) : (constraints)>,
  load_order = affine_map<(dimensions) -> (results)>,
  time_addr_map = affine_map<(time_dims, dimensions, symbols) -> (results)>,
  time_set = affine_set<(time_dims) : (time_constraints)>,
  time_order = affine_map<(time_dims) -> (results)>}
  {
  // Region body: operations for each time step, involving iter_data
  agen.yield
  } : memref_type
```

**Arguments**:
- `memref`: Base memory reference to load from, produced by dataflow.get_logical_memory_view
- `indices`: Variadic indices (element offsets) indicating starting location in memref
  (shown as `[%indices...]` in syntax, also called "subscripts map". %memref[%indices] is used as shown in agen.vector_load section.)
- `time_symbols(optional_time_symbols)`: in case the time_* parameters need symbols in addition to the indices for the time loops.
- `(iter_data : type)`: variable containing data loaded during current time step
- `dbgName` : optional debug name.
- `load_set`: Integer set defining load constraints, used as shown in agen.vector_load section
- `load_order`: Affine map defining load ordering, used as shown in agen.vector_load section
- `time_addr_map`: used to compute additional element offset for each time step (not affected by time_order) 
   added to the element offset computed from %memref[%indices], load_order, and layout_map.
- `time_set`: Indicates the bounds of each time loop.
- `time_order`:  Indicates the order of the time loops.


**Example**:
```
   #set0 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
   #map0 = affine_map<(d0) -> (d0)>

   #map1 = affine_map<(d0) -> (d0 * 64)>
   #set1 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

   %c0 = arith.constant 0 : index
   %c2048 = arith.constant 2048 : index

   %10 = dataflow.get_unit {core = 2 : i32, name = "l3lu", type = "l3lu"} : index
   %20 = dataflow.get_logical_memory_view %18, %c2048 {layout_map = #map0} : index, index, memref<64xf16>

   agen.composite_load %20[%c0 + %arg3 * 2048 + %arg4 * 2048 + %arg1 * 65536 + %arg2 * 32768]
     time_symbols() (%arg5:vector<64xf16>)
     {load_order = #map0, load_set = #set0, time_addr_map = #map1, time_order = #map0, time_set = #set1}
     {
      dataflow.send %10, %arg5 : vector<64xf16>
      agen.yield
     } : memref<64xf16>

pseudo-code:
-----------
   for d0 = 0, 31
      %arg5 = "load"
      // where "load" is similar to an agen.vector_load but with time_addr_map added to the 
      // source element offset which would have been calculated by agen.vector_load.
      //%arg5 = agen.vector_load %20[%c0 + %arg3 * 2048 + %arg4 * 2048 + %arg1 * 65536 + %arg2 * 32768] "+ time_addr_map"
      //                          { load_set, load_order} : memref<64xf16>
      // From composite_load region body:
      dataflow.send %10, %arg5 : vector<64xf16> 
      agen.yield
```

Traits: `MemRefsNormalizable`, `SingleBlockImplicitTerminator<YieldOp>`, `SingleBlock`

Interfaces: `AffineMapAccessInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>load_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>load_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>time_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>time_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>time_addr_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | memref of any type values
| `indices` | variadic of index


### `agen.composite_store` (agen::CompositeStoreOp)

_Agen composite store operation_

**Description**: agen.composite_store is equivalent to agen.vector_store invoked from within a loop nest, where the bounds and ordering of the loops are defined by time_set and time_order. An additional element offset from the base memref (coded immediately following "agen.composite_store") is specified via time_addr_map. Agen.composite_store contains a region body of the form 
```
      {
      %data = dataflow.receive %unit {} : vector<d0xd1xd2xi8>
      agen.yield %data : vector<1x1x1x128xi8>
      }
```
which specifies the vector source of the store for each time iteration. 

**Syntax**:
```mlir
agen.composite_store %memref[%indices...] 
  time_symbols(optional_time_symbols) {
  dbgName= "optional_debug_name",
  store_order = affine_map<(dimensions) -> (results)>,
  store_set = affine_set<(dimensions) : (constraints)>,
  time_addr_map = affine_map<(time_dims, dimensions, symbols) -> (results)>,
  time_set = affine_set<(time_dims) : (time_constraints)>,
  time_order = affine_map<(time_dims) -> (results)>}
  {
  // Region body: operations for each time step, involving iter_data
      %iter_data = dataflow.receive %unit {} : vector<1x1x1x128xi8>
      agen.yield %iter_data : vector<1x1x1x128xi8>
  } : memref_type
```

**Arguments**:
- `memref`: Base memory reference to load from, produced by dataflow.get_logical_memory_view
- `indices`: Variadic indices (element offsets) indicating starting location in memref
  (shown as `[%indices...]` in syntax, also called "subscripts map". %memref[%indices] is used as shown in agen.vector_store section.)
- `time_symbols(optional_time_symbols)`: in case the time_* parameters need symbols in addition to the indices for the time loops.
- `dbgName`: optional debug name.
- `store_order`: Affine map defining load ordering, used as shown in agen.vector_store section
- `store_set`: Integer set defining load constraints, used as shown in agen.vector_store section
- `time_addr_map`: used to compute additional element offset for each time step (not affected by time_order) 
   added to the element offset computed from %memref[%indices], store_order, and layout_map.
- `time_set`: Indicates the bounds of each time loop.
- `time_order`:  Indicates the order of the time loops.


**Example**:
```
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 128 + d1 * 4096 + d0 * 8192)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set = affine_set<(d0, d1, d2, d3) : (d3 >= 0, -d3 + 127 >= 0, d2 == 0, d1 == 0, d0 == 0)>

%c0 = arith.constant 0 : index 
%c3 = arith.constant 3 : index

%lx_c0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-CL0-LX", type="lx"} : index 
%lx_c0_mem_view = dataflow.get_logical_memory_view %lx_c0, %c0 {layout_map = #map1} : 
                                                   index, index, memref<?x?x?x128xi8>
%l3su_c1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "L3SU-C1-CL0", type = "l3su"} : index

agen.composite_store %lx_c0_mem_view[%arg0, %arg1, %arg2, %c0] time_symbols(%c3)
    {store_order = #map2,
     store_set = #set,
     time_set = affine_set<(d0)[s0]:(d0 >= 0, -d0 + s0 - 1 >= 0)>,
     time_order = affine_map<(d0) -> (d0)>,
     time_addr_map = affine_map<(d0) -> (0, 0, 0, 128 * d0)>}
     {
      %data = dataflow.receive %l3su_c1 {} : vector<1x1x1x128xi8>
      agen.yield %data :  vector<1x1x1x128xi8>
     } : memref<?x?x?x128xi8>


pseudo-code:
-----------
   for d0 = 0, 2 // s0 is 3 here, and s0 - 1 = 2
      %data = dataflow.receive %l3su_c1 {} : vector<1x1x1x128xi8>
      "store"
      // "store" is similar to an agen.vector_store, where time_addr_map is added to the target
      // element offset which would have been calculated by agen.vector_store.
      //agen.vector_store %data, %lx_c0_mem_view[%arg0, %arg1, %arg2, %c0] " + time_addr_map "
      //{store_order = #map2,
      // store_set = #set} : memref<?x?x?x128xi8>, vector<1x1x1x128xi8>
```

Traits: `MemRefsNormalizable`, `SingleBlockImplicitTerminator<YieldOp>`, `SingleBlock`

Interfaces: `AffineMapAccessInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>store_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>store_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>time_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>time_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>time_addr_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | memref of any type values
| `indices` | variadic of any type


### `agen.composite_load_and_store` (agen::CompositeLoadAndStoreOp)

_Agen composite load_and_store operation_

composite_load_and_store operations are used for loading from one memory
address and storing into another as part of each time step. For instance,
this operation is needed to model core-to-memory or memory-to-core data
transfers with burst in the L3 unit.

**Description**: agen.composite_load_and_store is a like a combined composite_load and composite_store, via a temporary vector, invoked from within a loop nest with "time steps". The bounds and ordering of the loops are defined by time_set and time_order. Additional element offsets from the base source and destination memrefs are specified via load_time_addr_map and store_time_order_map.

**Syntax**:
```mlir
agen.composite_load_and_store src:%memref_src[%indices_src...] dst:%memref_dst[%indices_dst...] 
  time_symbols(optional_time_symbols), (temp_vector : type), multicast_info = %multicast {
  dbgName = "optional_debug_name",
  load_set = affine_set<(dimensions) : (constraints)>,
  load_order = affine_map<(dimensions) -> (results)>,
  load_time_addr_map = affine_map<(time_dims, dimensions, symbols) -> (results)>,
  store_set = affine_set<(dimensions) : (constraints)>,
  store_order = affine_map<(dimensions) -> (results)>,
  store_time_addr_map = affine_map<(time_dims, dimensions, symbols) -> (results)>,
  time_set = affine_set<(time_dims) : (time_constraints)>,
  time_order = affine_map<(time_dims) -> (results)>}
  {
  // Region body: operations for each time step, involving temp_vector
  agen.yield
  } : memref_type

```

**Arguments**:
- `memref_src`: Base memory reference to load from, produced by dataflow.get_logical_memory_view
- `indices_src`: Variadic indices (element offsets) indicating starting location in memref_src
- `memref_dst`: Base memory reference to load from, produced by dataflow.get_logical_memory_view
- `indices_dst`: Variadic indices (element offsets) indicating starting location in memref_dst
  (Note: [%indices] are used as shown in agen.vector_load and vector_store sections.)
- `time_symbols(optional_time_symbols)`: in case the time_* parameters need symbols in addition to the indices for the time loops.
- `(temp_vector : type)`: temporary variable containing data loaded and stored during current time step
- `dbgName` : optional debug name.
- `load_set`: Integer set defining load constraints, used as shown in agen.vector_load section
- `load_order`: Affine map defining load ordering, used as shown in agen.vector_load section
- `load_time_addr_map`: used to compute additional element offset for each time step (not affected by time_order)
   added to the element offset computed from %memref_src[%indices_src], load_order, and source layout_map.
- `store_set`: Integer set defining load constraints, used as shown in agen.vector_store section
- `store_order`: Affine map defining load ordering, used as shown in agen.vector_store section
- `store_time_addr_map`: used to compute additional element offset for each time step (not affected by time_order)
   added to the element offset computed from %memref_dst[%indices_dst], load_order, and destination layout_map.
- `time_set`: Indicates the bounds of each time loop.
- `time_order`:  Indicates the order of the time loops.

**Example**:
```
  #map = affine_map<(d0) -> (d0)>
  #set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
  #set1 = affine_set<(d0) : (d0 >= 0, -d0 >= 0)>

  %c153600 = arith.constant 153600 : index
  %c0_3 = arith.constant 0 : index
  %8 = dataflow.get_logical_memory_view %lx, %c153600 {layout_map = #map} : index, index, memref<32xi32>
  %c0_4 = arith.constant 0 : index
  %9 = dataflow.get_logical_memory_view %ibr, %c0_4 {layout_map = #map} : index, index, memref<32xi32>
  %c0_5 = arith.constant 0 : index

  agen.composite_load_and_store src:%8[%c0_3 + %arg0 * 32 + %arg2 * 32] dst:%9[%c0_5]
        time_symbols(), load_iv(%arg5:vector<32xi32>)
        {load_order = #map, load_set = #set, load_time_addr_map = #map, 
         store_order = #map, store_set = #set, store_time_addr_map = #map, 
         time_order = #map, time_set = #set1}
        {
           agen.yield
        } : memref<32xi32>, memref<32xi32>


pseudo-code:
-----------
   agen.composite_load %8[%c0_3 + %arg0 * 32 + %arg2 * 32]
     time_symbols() load_iv(%arg5:vector<32xi32>)
     {load_order = #map, load_set = #set, load_time_addr_map = #map, time_order = #map, time_set = #set1}
     {
      "store"
      // where "store" is similar to agen.vector_store but with store_time_addr_map 
      // added to the target element offset which vector_store would normally compute.
      // agen.vector_store %arg5, %9[%c0_5] "+ store_time_addr_map" 
      //               {store_order = #map, store_set = #set} : memref<32xi32>, vector<32xi32>
      agen.yield 
     } :  memref<32xi32>
```

Traits: `MemRefsNormalizable`, `SingleBlockImplicitTerminator<YieldOp>`, `SingleBlock`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dbgName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>load_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>load_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>store_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>store_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>time_set</code></td><td>::mlir::IntegerSetAttr</td><td>An Attribute containing an IntegerSet object</td></tr>
<tr><td><code>time_order</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>load_time_addr_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>store_time_addr_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
<tr><td><code>num_time_symbols</code></td><td>::mlir::IntegerAttr</td><td>32-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref_src` | memref of any type values
| `memref_dst` | memref of any type values
| `indices_src` | variadic of index
| `indices_dst` | variadic of index

