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
// This file defines the operations in the KTDF operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DT_EXCEPTION_HPP
#define DT_EXCEPTION_HPP

#include <cassert>

#define DT_CHECK(testable)                               \
  do {                                                   \
    const auto _condition = static_cast<bool>(testable); \
    assert(_condition);                                  \
  } while (0)

#define DT_CHECK_MSG(testable, message)                  \
  do {                                                   \
    const auto _condition = static_cast<bool>(testable); \
    assert(_condition && message);                       \
  } while (0)

#endif  // DT_EXCEPTION_HPP
