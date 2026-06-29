// RUN: dataflow-scheduler --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-DAG: ktdp
// CHECK-DAG: ktdf
// CHECK-DAG: ktdf_arch
// CHECK-DAG: agen
// CHECK-DAG: dataflow
// CHECK-DAG: uniform
// CHECK-DAG: vectorchain
