// RUN: dataflow-scheduler-opt --show-dialects | FileCheck %s

// CHECK: Available Dialects:

// CHECK-DAG: affine
// CHECK-DAG: arith
// CHECK-DAG: func
// CHECK-DAG: linalg
// CHECK-DAG: math
// CHECK-DAG: memref
// CHECK-DAG: scf
// CHECK-DAG: tensor

// CHECK-DAG: ktdp

// CHECK-DAG: agen
// CHECK-DAG: dataflow
// CHECK-DAG: ktdf
// CHECK-DAG: ktdf_arch
// CHECK-DAG: ktdf_lowering
// CHECK-DAG: uniform
// CHECK-DAG: vectorchain
