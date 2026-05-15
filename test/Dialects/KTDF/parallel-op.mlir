// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   func.func @parallel_1d_unit_step
// CHECK:           ktdf.parallel (%[[M:.*]], %[[INST:.*]]) = (%[[C0:.*]]) to (%[[C64:.*]]) step (%[[C1:.*]]) distribute(num_instances = 2) {
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }

func.func @parallel_1d_unit_step() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  ktdf.parallel (%m, %inst) = (%c0) to (%c64) step (%c1)
                              distribute(num_instances = 2) {
    ktdf.parallel_yield
  }
  return
}

// CHECK-LABEL:   func.func @parallel_2d
// CHECK:           ktdf.parallel (%[[P:.*]], %[[Q:.*]], %[[INST:.*]]) = (%[[C0:.*]], %[[C0_1:.*]]) to (%[[C5:.*]], %[[C7:.*]]) step (%[[C1:.*]], %[[C1_1:.*]]) distribute(num_instances = 2) {
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }

func.func @parallel_2d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  ktdf.parallel (%p, %q, %inst) = (%c0, %c0) to (%c5, %c7) step (%c1, %c1)
                                  distribute(num_instances = 2) {
    ktdf.parallel_yield
  }
  return
}

// CHECK-LABEL:   func.func @parallel_non_unit_step
// CHECK:           ktdf.parallel (%[[I:.*]], %[[INST:.*]]) = (%[[C0:.*]]) to (%[[C32:.*]]) step (%[[C2:.*]]) distribute(num_instances = 2) {
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }

func.func @parallel_non_unit_step() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  ktdf.parallel (%i, %inst) = (%c0) to (%c32) step (%c2)
                              distribute(num_instances = 2) {
    ktdf.parallel_yield
  }
  return
}

// CHECK-LABEL:   func.func @parallel_uses_instance_id
// CHECK:           ktdf.parallel (%[[I:.*]], %[[INST:[a-zA-Z0-9_]+]]) = (%[[C0:.*]]) to (%[[C64:.*]]) step (%[[C1:.*]]) distribute(num_instances = 2) {
// CHECK-NEXT:        "test.use"(%[[INST]]) : (index) -> ()
// CHECK-NEXT:        ktdf.parallel_yield
// CHECK-NEXT:      }

func.func @parallel_uses_instance_id() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  ktdf.parallel (%i, %inst) = (%c0) to (%c64) step (%c1)
                              distribute(num_instances = 2) {
    "test.use"(%inst) : (index) -> ()
    ktdf.parallel_yield
  }
  return
}

// CHECK-LABEL:   func.func @parallel_inside_coarsened_stage
// CHECK:           ktdf.pipeline {
// CHECK:             ktdf.stage depends_in(%{{.*}} : !ktdf.token) depends_out(%{{.*}} : !ktdf.token) {
// CHECK:               ktdf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) distribute(num_instances = 2) {
// CHECK:                 ktdf.parallel_yield
// CHECK:               }
// CHECK:             }
// CHECK:           }

func.func @parallel_inside_coarsened_stage() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  %t1 = ktdf.create_token : !ktdf.token
  %t5 = ktdf.create_token : !ktdf.token

  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
      "test.stage0_body"() : () -> ()
    }

    ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t5 : !ktdf.token) {
      ktdf.parallel (%m, %n, %inst) = (%c0, %c0) to (%c32, %c1) step (%c1, %c1)
                                      distribute(num_instances = 2) {
        ktdf.parallel_yield
      }
    }

    ktdf.stage depends_in(%t5 : !ktdf.token) depends_out(none) {
      "test.stage4_body"() : () -> ()
    }
  }
  return
}
