//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "Ktdp-c/Dialects.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/Tensor.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "src/CAPI/KTDF.hpp"

namespace nb = nanobind;

NB_MODULE(_ktir, m) {
  m.doc() = "mlir_ktir main python extension";

  m.def(
      "register_dialects",
      [](MlirContext context, bool load) {
        for (auto handle :
             {mlirGetDialectHandle__arith__(), mlirGetDialectHandle__func__(),
              mlirGetDialectHandle__linalg__(), mlirGetDialectHandle__math__(),
              mlirGetDialectHandle__scf__(), mlirGetDialectHandle__tensor__(),
              mlirGetDialectHandle__ktdp__(), mlirGetDialectHandle__ktdf__()}) {
          mlirDialectHandleRegisterDialect(handle, context);
          if (load) mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
}