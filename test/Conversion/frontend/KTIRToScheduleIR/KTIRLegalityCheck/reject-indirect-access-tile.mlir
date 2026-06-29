// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s

#coord_set_1d_8 = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
#coord_set_2d_100x16 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 99 >= 0, d1 >= 0, -d1 + 15 >= 0)>
#var_space_2d_8x16 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 15 >= 0)>
#order_2d_identity = affine_map<(d0, d1) -> (d0, d1)>

func.func @indirect_access_tile() {
  %addr_x = arith.constant 10000 : index
  %addr_idx = arith.constant 20000 : index

  %idx_view = ktdp.construct_memory_view %addr_idx,
      sizes: [8], strides: [1] {
      coordinate_set = #coord_set_1d_8,
      memory_space = #ktdp.spyre_memory_space<HBM>
  } : memref<8xi32>

  %x_view = ktdp.construct_memory_view %addr_x,
      sizes: [100, 16], strides: [16, 1] {
      coordinate_set = #coord_set_2d_100x16,
      memory_space = #ktdp.spyre_memory_space<HBM>
  } : memref<100x16xf16>

  // expected-error @+1 {{V1 does not support ktdp.construct_indirect_access_tile}}
  %tile = ktdp.construct_indirect_access_tile
      intermediate_variables(%i, %j)
      %x_view[(%i), ind(%idx_view[%j])] {
      variables_space_set = #var_space_2d_8x16,
      variables_space_order = #order_2d_identity
  } : memref<100x16xf16>, memref<8xi32> -> !ktdp.access_tile<8x16xindex>

  return
}
