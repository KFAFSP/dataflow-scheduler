// RUN: dataflow-scheduler-opt -normalize-grid-to-1d -split-input-file -verify-diagnostics %s

// Negative: zero dimension.
func.func @grid_zero_dim() attributes {grid = [2 : index, 0 : index]} {
  // expected-error@-1 {{Grid size must be positive (got 0)}}
  %0:2 = ktdp.get_compute_tile_id : index, index
  return
}

// -----

// Negative: empty grid.
func.func @grid_empty() attributes {grid = []} {
  // expected-error@-1 {{Grid must have at least one dimension}}
  return
}
