// RUN: scheduler %s | scheduler | FileCheck %s

// Round trip test for ktdf.create_token operation

// CHECK-LABEL:   func.func @create_token_basic() {
// CHECK-NEXT:     %[[VAL_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @create_token_basic() {
    %token = ktdf.create_token : !ktdf.token
    return
  }
}