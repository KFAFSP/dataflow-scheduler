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

#include <mlir-c/IR.h>
#include <mlir/CAPI/Support.h>

#include <cstdint>
#include <cstdio>

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "src/CAPI/RegisterEverything.h"

using namespace mlir;
namespace nb = nanobind;

NB_MODULE(_ktir, m) {
  schedulerRegisterAllPasses();

  m.doc() = "mlir_ktir main python extension";

  m.def(
      "register_dialects",
      [](MlirDialectRegistry registry) {
        schedulerRegisterAllDialects(registry);
      },
      nb::arg("registry"));
}