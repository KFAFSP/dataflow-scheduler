//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Conversion/Utils/Utils.h"

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"

namespace scheduler {

auto getUnitResourceType(mlir::Value unit_value)
    -> std::optional<scheduler::ResourceType> {
  auto query_op = unit_value.getDefiningOp<mlir::uniform::QueryMapOp>();
  if (!query_op) {
    return std::nullopt;
  }

  auto def_mapping_op =
      query_op.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
  if (!def_mapping_op) {
    return std::nullopt;
  }

  auto values = def_mapping_op.getValues();
  if (values.empty()) {
    return std::nullopt;
  }

  auto get_unit_op = values.front().getDefiningOp<mlir::dataflow::GetUnitOp>();
  if (!get_unit_op) {
    return std::nullopt;
  }

  return mlir::StringAttr::get(unit_value.getContext(),
                               get_unit_op.getType().upper());
}

std::string getUnitTypeFromQueryMap(mlir::Value query_map) {
  // Get the query_map operation
  auto query_op = query_map.getDefiningOp<mlir::uniform::QueryMapOp>();
  if (!query_op) {
    query_map.getDefiningOp()->emitError(
        "expected query_map to be defined by uniform.query_map operation");
    return "";
  }

  // Get the def_immutable_mapping
  auto def_map_op =
      query_op.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
  if (!def_map_op) {
    query_op->emitError(
        "expected query_map's map to be defined by "
        "uniform.def_immutable_mapping operation");
    return "";
  }

  // Get the first value from the mapping (first result unit)
  auto values = def_map_op.getValues();
  if (values.empty()) {
    def_map_op->emitError(
        "expected def_immutable_mapping to have at least one value");
    return "";
  }

  mlir::Value first_unit = values[0];

  // Get the dataflow.get_unit operation
  auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
      first_unit.getDefiningOp());
  if (!get_unit) {
    first_unit.getDefiningOp()->emitError(
        "expected first unit value to be defined by dataflow.get_unit "
        "operation");
    return "";
  }

  // Extract the type using getType().upper()
  std::string type_str = get_unit.getType().upper();
  return type_str;
}

}  // namespace scheduler

// Made with Bob
