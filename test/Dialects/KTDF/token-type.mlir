// RUN: scheduler -allow-unregistered-dialect %s | scheduler -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   func.func @token_type_test() {
// CHECK-NEXT:     %[[VAL_0:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:     %[[VAL_1:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:     %[[VAL_2:.*]] = "test.op"(%[[VAL_0]]) : (!ktdf.token) -> !ktdf.token
// CHECK-NEXT:     %[[VAL_3:.*]] = "test.op"(%[[VAL_0]], %[[VAL_1]]) : (!ktdf.token, !ktdf.token) -> !ktdf.token
// CHECK-NEXT:     "test.op"(%[[VAL_2]], %[[VAL_3]]) : (!ktdf.token, !ktdf.token) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @token_type_test() {
    // Create token values
    %0 = "test.op"() : () -> !ktdf.token
    %1 = "test.op"() : () -> !ktdf.token
    
    // Use tokens for dataflow dependencies
    %2 = "test.op"(%0) : (!ktdf.token) -> !ktdf.token
    %3 = "test.op"(%0, %1) : (!ktdf.token, !ktdf.token) -> !ktdf.token
    
    // Consume tokens
    "test.op"(%2, %3) : (!ktdf.token, !ktdf.token) -> ()
    
    return
  }
}
