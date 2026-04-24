// RUN: scheduler %s --kEmitDFIR | FileCheck %s

// Test that SCF parallel loop tiling is applied in the KTDP to DFIR pipeline

func.func @test_parallel_loop_tiling() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  
  // CHECK: scf.parallel
  // CHECK: scf.parallel
  scf.parallel (%i, %j) = (%c0, %c0) to (%c100, %c100) step (%c1, %c1) {
    // Empty body for testing - scf.parallel doesn't need explicit terminator
  }
  
  return
}