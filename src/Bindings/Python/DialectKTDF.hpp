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
//
// This file provides traits for nanobind to allow type conversion to auto-
// generated Python types, e.g., int-based enums.
//
//===----------------------------------------------------------------------===//

#ifndef BINDINGS_PYTHON_DIALECTKTDF_HPP
#define BINDINGS_PYTHON_DIALECTKTDF_HPP

#include <mlir-c/Bindings/Python/Interop.h>
#include <nanobind/nanobind.h>

#include <type_traits>

#include "src/Dialects/KTDF/KTDFTypes.hpp"

namespace nb = nanobind;

#define IMPORT_INT_ENUM_TYPECASTER(name)                                      \
  template <>                                                                 \
  struct nb::detail::type_caster<mlir::ktdf::name> {                          \
    NB_TYPE_CASTER(mlir::ktdf::name, const_name(MAKE_MLIR_PYTHON_QUALNAME(    \
                                         "dialects.ktdf." #name)))            \
                                                                              \
    bool from_python(handle src, uint8_t, cleanup_list*) noexcept {           \
      std::underlying_type_t<mlir::ktdf::name> as_int;                        \
      if (!nb::try_cast(src, as_int)) return false;                           \
      value = static_cast<mlir::ktdf::name>(as_int);                          \
      return true;                                                            \
    }                                                                         \
                                                                              \
    static handle from_cpp(mlir::ktdf::name src, rv_policy,                   \
                           cleanup_list*) noexcept {                          \
      const auto as_int =                                                     \
          static_cast<std::underlying_type_t<mlir::ktdf::name>>(src);         \
                                                                              \
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("dialects.ktdf")) \
          .attr(#name)(as_int)                                                \
          .release();                                                         \
    }                                                                         \
  }

IMPORT_INT_ENUM_TYPECASTER(ComponentType);
IMPORT_INT_ENUM_TYPECASTER(FifoType);
IMPORT_INT_ENUM_TYPECASTER(LoopType);
IMPORT_INT_ENUM_TYPECASTER(MemorySpace);

#undef IMPORT_I32_ENUM_TYPECASTER

#endif  // BINDINGS_PYTHON_DIALECTKTDF_HPP
