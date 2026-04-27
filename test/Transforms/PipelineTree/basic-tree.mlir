// RUN: scheduler %s -print-pipeline-tree | FileCheck %s

// This test verifies that we can parse a basic pipeline structure
// The PipelineTree abstraction will be used to analyze this structure

func.func @simple_pipeline() {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  
  %token0 = ktdf.create_token : !ktdf.token
  
  // Outer loop
  scf.for %i = %c0 to %c100 step %c1 {
    // Pipeline with two stages
    ktdf.pipeline {
      // Stage 0
      ktdf.stage depends_in(none) depends_out(%token0 : !ktdf.token) {
        // Some computation
        %val = arith.constant 42 : i32
      }
      
      // Stage 1
      ktdf.stage depends_in(%token0 : !ktdf.token) depends_out(none) {
        // Some computation
        %val = arith.constant 43 : i32
      }
    }
  }
  
  return
}

// CHECK-LABEL: === Pipeline Tree for function: simple_pipeline ===
// CHECK: loop
// CHECK:   pipeline
// CHECK:     stage 0
// CHECK:     stage 1
// CHECK: === End Pipeline Tree ===
