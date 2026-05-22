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

#ifndef KTDF_CAPI_H
#define KTDF_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all KTIR scheduler dialects and extensions to the dialect registry.
MLIR_CAPI_EXPORTED void schedulerRegisterAllDialects(
    MlirDialectRegistry registry);

/// Register all compiler passes of the KTIR scheduler.
MLIR_CAPI_EXPORTED void schedulerRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif  // KTDF_CAPI_H