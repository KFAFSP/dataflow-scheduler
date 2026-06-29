//===-- dt_exception.h -------------------------------------------*- c++ -*-==//
//
// Part of the Dataflow Scheduler project.
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
// This file defines the DT_CHECK and DT_CHECK_MSG macros from DeepTools.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DT_EXCEPTION_H_
#define DATAFLOW_SCHEDULER_DT_EXCEPTION_H_

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

#endif  // DATAFLOW_SCHEDULER_DT_EXCEPTION_H_
