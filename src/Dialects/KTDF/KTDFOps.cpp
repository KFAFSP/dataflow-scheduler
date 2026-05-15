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
#include "KTDFOps.hpp"

#include <iostream>
#include <iterator>
#include <regex>

#include "KTDFTypes.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Dialects/KTDF/KTDFEnums.cpp.inc"
#include "src/Dialects/KTDF/KTDFOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// KTDFDialect
//===----------------------------------------------------------------------===//

void KTDFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Dialects/KTDF/KTDFOps.cpp.inc"

      >();
  registerTypes();
  registerAttributes();
}

//===----------------------------------------------------------------------===//
// Custom Parsers and Printers
//===----------------------------------------------------------------------===//

// Custom parser for sizes list that supports both SSA values and constants
static ParseResult parseSizesList(
    OpAsmParser& parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand>& sizes,
    DenseI64ArrayAttr& static_sizes) {
  SmallVector<int64_t> static_size_values;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamic_sizes;

  auto parse_elt = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand operand;
    auto res = parser.parseOptionalOperand(operand);
    if (res.has_value() && succeeded(res.value())) {
      // It's an SSA value
      dynamic_sizes.push_back(operand);
      static_size_values.push_back(ShapedType::kDynamic);
      return success();
    }

    // Try to parse as integer constant
    int64_t value;
    if (succeeded(parser.parseInteger(value))) {
      static_size_values.push_back(value);
      return success();
    }

    return failure();
  };

  if (parser.parseLSquare() || parser.parseCommaSeparatedList(parse_elt) ||
      parser.parseRSquare())
    return failure();

  // Set the results
  sizes.append(dynamic_sizes.begin(), dynamic_sizes.end());
  if (!static_size_values.empty()) {
    static_sizes = parser.getBuilder().getDenseI64ArrayAttr(static_size_values);
  }

  return success();
}

// Custom printer for sizes list
static void printSizesList(OpAsmPrinter& printer, Operation* op,
                           OperandRange sizes, DenseI64ArrayAttr static_sizes) {
  printer << "[";

  if (!static_sizes) {
    // All dynamic - print SSA values
    llvm::interleaveComma(sizes, printer);
  } else {
    // Mix of static and dynamic
    ArrayRef<int64_t> static_values = static_sizes;
    unsigned dynamic_idx = 0;

    llvm::interleaveComma(static_values, printer, [&](int64_t value) {
      if (ShapedType::isDynamic(value)) {
        printer << sizes[dynamic_idx++];
      } else {
        printer << value;
      }
    });
  }

  printer << "]";
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialects/KTDF/KTDFOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PipelineOp
//===----------------------------------------------------------------------===//

PipelineOp::StageRange PipelineOp::getStages() {
  return getBodyBlock()->getOps<StageOp>();
}

unsigned PipelineOp::getNumStages() {
  auto stages = getStages();
  return std::distance(stages.begin(), stages.end());
}

//===----------------------------------------------------------------------===//
// FifoAllocateOp
//===----------------------------------------------------------------------===//

LogicalResult FifoAllocateOp::verify() {
  auto dynamic_sizes = getDynamicSizes();
  auto slots = getSlots();

  // Count the number of dynamic sizes in the result types
  unsigned num_dynamic_sizes = 0;
  for (auto slot : slots) {
    auto slot_type = mlir::cast<FifoSlotType>(slot.getType());
    if (slot_type.isDynamicNumElements()) {
      num_dynamic_sizes++;
    }
  }

  // Verify that the number of dynamic size operands matches the number of
  // dynamic sizes in the result types
  if (dynamic_sizes.size() != num_dynamic_sizes) {
    return emitOpError("number of dynamic size operands (")
           << dynamic_sizes.size()
           << ") must match number of dynamic sizes in result types ("
           << num_dynamic_sizes << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PipelineOp
//===----------------------------------------------------------------------===//

LogicalResult PipelineOp::verify() {
  // Verify that all immediate children are StageOp or PrivateOp
  Block* body = getBodyBlock();
  for (Operation& op : body->getOperations()) {
    if (!isa<StageOp, PrivateOp>(op)) {
      return emitOpError(
                 "immediate children of pipeline must be ktdf.stage or "
                 "ktdf.private operations, found '")
             << op.getName() << "'";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

ParseResult StageOp::parse(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::UnresolvedOperand> depends_in;
  SmallVector<Type> depends_in_types;
  SmallVector<OpAsmParser::UnresolvedOperand> depends_out;
  SmallVector<Type> depends_out_types;

  // Parse depends_in
  if (succeeded(parser.parseOptionalKeyword("depends_in"))) {
    if (parser.parseLParen()) return failure();

    if (failed(parser.parseOptionalKeyword("none"))) {
      // Parse token operands
      if (parser.parseOperandList(depends_in) || parser.parseColon() ||
          parser.parseTypeList(depends_in_types))
        return failure();
    }

    if (parser.parseRParen()) return failure();
  }

  // Parse depends_out
  if (succeeded(parser.parseOptionalKeyword("depends_out"))) {
    if (parser.parseLParen()) return failure();

    if (failed(parser.parseOptionalKeyword("none"))) {
      // Parse token operands
      if (parser.parseOperandList(depends_out) || parser.parseColon() ||
          parser.parseTypeList(depends_out_types))
        return failure();
    }

    if (parser.parseRParen()) return failure();
  }

  // Resolve operands
  if (parser.resolveOperands(depends_in, depends_in_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  if (parser.resolveOperands(depends_out, depends_out_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  // Parse the body region
  Region* body = result.addRegion();
  if (parser.parseRegion(*body)) return failure();

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  // Add operand segment sizes attribute if not already present.
  // The attribute is required since op has multiple optional/variadic
  // operands.
  if (!result.attributes.get("operandSegmentSizes")) {
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(depends_in.size()),
                             static_cast<int32_t>(depends_out.size())}));
  }

  return success();
}

void StageOp::print(OpAsmPrinter& p) {
  // Print depends_in
  p << " depends_in(";
  if (getDependsIn().empty()) {
    p << "none";
  } else {
    p.printOperands(getDependsIn());
    p << " : ";
    llvm::interleaveComma(getDependsIn(), p,
                          [&](Value operand) { p << operand.getType(); });
  }
  p << ")";

  // Print depends_out
  p << " depends_out(";
  if (getDependsOut().empty()) {
    p << "none";
  } else {
    p.printOperands(getDependsOut());
    p << " : ";
    llvm::interleaveComma(getDependsOut(), p,
                          [&](Value operand) { p << operand.getType(); });
  }
  p << ")";

  // Print the body region
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  // Print attributes.
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static bool isValidTokenSource(Value token_value);

LogicalResult StageOp::verify() {
  // Verify that all depends_in tokens come from valid sources
  for (Value token : getDependsIn()) {
    if (!isValidTokenSource(token)) {
      return emitOpError(
          "depends_in operand must originate from ktdf.create_token or "
          "ktdf.private operation yielding a token created by "
          "ktdf.create_token");
    }
  }

  // Verify that all depends_out tokens come from valid sources
  for (Value token : getDependsOut()) {
    if (!isValidTokenSource(token)) {
      return emitOpError(
          "depends_out operand must originate from ktdf.create_token or "
          "ktdf.private operation yielding a token created by "
          "ktdf.create_token");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageOp Verification Helper
//===----------------------------------------------------------------------===//

// Helper function to check if a value originates from create_token or private
// operation yielding a token
static bool isValidTokenSource(Value token_value) {
  Operation* defining_op = token_value.getDefiningOp();
  if (!defining_op) {
    return false;
  }

  // Check if it's directly from create_token
  if (isa<CreateTokenOp>(defining_op)) {
    return true;
  }

  // Check if it's from a private operation
  if (auto private_op = dyn_cast<PrivateOp>(defining_op)) {
    // Get the result index
    auto result_index = llvm::cast<OpResult>(token_value).getResultNumber();

    // Get the corresponding yield operand
    auto yield_op = private_op.getYieldOp();
    if (result_index >= yield_op.getOperands().size()) {
      return false;
    }

    Value yield_operand = yield_op.getOperands()[result_index];

    // Walk up the chain to find the ultimate source
    // The yielded value must trace back to a create_token
    Operation* yield_source = yield_operand.getDefiningOp();
    if (!yield_source) {
      return false;
    }

    return isa<CreateTokenOp>(yield_source);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// PrivateOp
//===----------------------------------------------------------------------===//

PrivateYieldOp PrivateOp::getYieldOp() {
  return cast<PrivateYieldOp>(getBodyBlock()->getTerminator());
}

LogicalResult PrivateOp::verify() {
  // Verify that the region has a terminator
  Block* body = getBodyBlock();
  if (body->empty() || !body->back().hasTrait<OpTrait::IsTerminator>()) {
    return emitOpError("region must be terminated with ktdf.private_yield");
  }

  // Verify that the terminator is a PrivateYieldOp
  auto yield_op = dyn_cast<PrivateYieldOp>(body->getTerminator());
  if (!yield_op) {
    return emitOpError("region must be terminated with ktdf.private_yield");
  }

  // Verify that the number and types of yield operands match the results
  auto yield_operands = yield_op.getOperands();
  auto results = getResults();

  if (yield_operands.size() != results.size()) {
    return emitOpError("yield operands count (")
           << yield_operands.size() << ") must match results count ("
           << results.size() << ")";
  }

  for (auto [yield_operand, result] : llvm::zip(yield_operands, results)) {
    if (yield_operand.getType() != result.getType()) {
      return emitOpError("yield operand type ")
             << yield_operand.getType() << " must match result type "
             << result.getType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DataTransferOp
//===----------------------------------------------------------------------===//

LogicalResult DataTransferOp::verify() {
  auto source_type = getSource().getType();
  auto dest_type = getDestination().getType();

  // Verify that source and destination are not both FIFOs
  bool source_fifo = llvm::isa<FifoSlotType>(source_type);
  bool dest_fifo = llvm::isa<FifoSlotType>(dest_type);

  if (source_fifo && dest_fifo) {
    return emitOpError("cannot transfer directly from FIFO to FIFO");
  }

  // Verify that at least one is a memref or FIFO
  bool source_memref = llvm::isa<MemRefType>(source_type);
  bool dest_memref = llvm::isa<MemRefType>(dest_type);

  if (!source_memref && !source_fifo) {
    return emitOpError("source must be a memref or FIFO slot");
  }

  if (!dest_memref && !dest_fifo) {
    return emitOpError("destination must be a memref or FIFO slot");
  }

  // Verify that FIFO slots have no indices specified
  if (source_fifo && !getSourceIndices().empty()) {
    return emitOpError("FIFO slot source cannot have indices specified");
  }

  if (dest_fifo && !getDestIndices().empty()) {
    return emitOpError("FIFO slot destination cannot have indices specified");
  }

  // Get element types for compatibility checking
  Type source_elem_type;
  Type dest_elem_type;

  if (source_memref) {
    source_elem_type = llvm::cast<MemRefType>(source_type).getElementType();
  } else if (source_fifo) {
    source_elem_type = llvm::cast<FifoSlotType>(source_type).getElementType();
  }

  if (dest_memref) {
    dest_elem_type = llvm::cast<MemRefType>(dest_type).getElementType();
  } else if (dest_fifo) {
    dest_elem_type = llvm::cast<FifoSlotType>(dest_type).getElementType();
  }

  // Verify element type compatibility
  if (source_elem_type && dest_elem_type &&
      source_elem_type != dest_elem_type) {
    return emitOpError("source and destination element types must match, got ")
           << source_elem_type << " and " << dest_elem_type;
  }

  // Verify indices and sizes are consistent for memrefs
  auto source_memref_type = llvm::dyn_cast<MemRefType>(source_type);
  if (source_memref_type) {
    unsigned rank = source_memref_type.getRank();
    if (!getSourceIndices().empty() && getSourceIndices().size() != rank) {
      return emitOpError("source indices count (")
             << getSourceIndices().size() << ") must match source memref rank ("
             << rank << ")";
    }
  }

  auto dest_memref_type = llvm::dyn_cast<MemRefType>(dest_type);
  if (dest_memref_type) {
    unsigned rank = dest_memref_type.getRank();
    if (!getDestIndices().empty() && getDestIndices().size() != rank) {
      return emitOpError("destination indices count (")
             << getDestIndices().size()
             << ") must match destination memref rank (" << rank << ")";
    }
  }

  // Verify that static and dynamic sizes are consistent
  auto verifySize = [&](OperandRange dynamicSizes,
                        std::optional<ArrayRef<int64_t>> static_sizes,
                        StringRef name) -> LogicalResult {
    if (!static_sizes) {
      // All sizes are dynamic - no verification needed
      return success();
    }

    ArrayRef<int64_t> static_values = *static_sizes;
    unsigned num_dynamic = llvm::count_if(
        static_values, [](int64_t val) { return ShapedType::isDynamic(val); });

    if (dynamicSizes.size() != num_dynamic) {
      return emitOpError("number of dynamic ")
             << name << " sizes (" << dynamicSizes.size()
             << ") must match number of dynamic entries in static_" << name
             << "_sizes (" << num_dynamic << ")";
    }

    return success();
  };

  if (failed(verifySize(getSourceSizes(), getStaticSourceSizes(), "source")))
    return failure();

  if (failed(verifySize(getDestSizes(), getStaticDestSizes(), "dest")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ReadFromFifoOp
//===----------------------------------------------------------------------===//

LogicalResult ReadFromFifoOp::verify() {
  auto fifo_slot_type = llvm::cast<FifoSlotType>(getFifoSlot().getType());
  auto result_type = llvm::dyn_cast<RankedTensorType>(getResult().getType());

  if (!result_type) {
    return emitOpError("result must be a ranked tensor type");
  }

  // Verify element types match
  if (fifo_slot_type.getElementType() != result_type.getElementType()) {
    return emitOpError("result tensor element type (")
           << result_type.getElementType()
           << ") must match FIFO slot element type ("
           << fifo_slot_type.getElementType() << ")";
  }

  // Verify total number of elements matches FIFO slot size
  int64_t fifo_num_elements = fifo_slot_type.getNumElements();
  int64_t tensor_num_elements = result_type.getNumElements();

  // Both must be dynamic or both must be static with matching values
  bool fifo_dynamic = ShapedType::isDynamic(fifo_num_elements);
  bool tensor_dynamic = ShapedType::isDynamic(tensor_num_elements);

  if (fifo_dynamic != tensor_dynamic) {
    return emitOpError(
        "FIFO slot and result tensor must both be dynamic or both be static");
  }

  if (!fifo_dynamic && fifo_num_elements != tensor_num_elements) {
    return emitOpError("result tensor total elements (")
           << tensor_num_elements << ") must match FIFO slot size ("
           << fifo_num_elements << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// WriteToFifoOp
//===----------------------------------------------------------------------===//

LogicalResult WriteToFifoOp::verify() {
  auto fifo_slot_type = llvm::cast<FifoSlotType>(getFifoSlot().getType());
  auto data_type = llvm::dyn_cast<RankedTensorType>(getData().getType());

  if (!data_type) {
    return emitOpError("data operand must be a ranked tensor type");
  }

  // Verify element types match
  if (fifo_slot_type.getElementType() != data_type.getElementType()) {
    return emitOpError("data tensor element type (")
           << data_type.getElementType()
           << ") must match FIFO slot element type ("
           << fifo_slot_type.getElementType() << ")";
  }

  // Verify total number of elements matches FIFO slot size
  int64_t fifo_num_elements = fifo_slot_type.getNumElements();
  int64_t tensor_num_elements = data_type.getNumElements();

  // Both must be dynamic or both must be static with matching values
  bool fifo_dynamic = ShapedType::isDynamic(fifo_num_elements);
  bool tensor_dynamic = ShapedType::isDynamic(tensor_num_elements);

  if (fifo_dynamic != tensor_dynamic) {
    return emitOpError(
        "FIFO slot and data tensor must both be dynamic or both be static");
  }

  if (!fifo_dynamic && fifo_num_elements != tensor_num_elements) {
    return emitOpError("data tensor total elements (")
           << tensor_num_elements << ") must match FIFO slot size ("
           << fifo_num_elements << ")";
  }

  return success();
}

// Build with mixed static/dynamic sizes (OpFoldResult can be either Attribute
// or Value)
void DataTransferOp::build(OpBuilder& builder, OperationState& result,
                           Value source, ValueRange source_indices,
                           ArrayRef<OpFoldResult> source_sizes,
                           Value destination, ValueRange dest_indices,
                           ArrayRef<OpFoldResult> dest_sizes) {
  SmallVector<int64_t> static_source_sizes, static_dest_sizes;
  SmallVector<Value> dynamic_source_sizes, dynamic_dest_sizes;

  // Dispatch OpFoldResults into static attributes and dynamic values
  dispatchIndexOpFoldResults(source_sizes, dynamic_source_sizes,
                             static_source_sizes);
  dispatchIndexOpFoldResults(dest_sizes, dynamic_dest_sizes, static_dest_sizes);

  // Build the operation with separated static and dynamic components
  build(builder, result, source, source_indices, dynamic_source_sizes,
        destination, dest_indices, dynamic_dest_sizes,
        builder.getDenseI64ArrayAttr(static_source_sizes),
        builder.getDenseI64ArrayAttr(static_dest_sizes));
}

// Build with all static sizes
void DataTransferOp::build(OpBuilder& builder, OperationState& result,
                           Value source, ValueRange source_indices,
                           ArrayRef<int64_t> source_sizes, Value destination,
                           ValueRange dest_indices,
                           ArrayRef<int64_t> dest_sizes) {
  // Convert static sizes to OpFoldResult (as attributes)
  SmallVector<OpFoldResult> source_size_values = llvm::map_to_vector<4>(
      source_sizes,
      [&](int64_t v) -> OpFoldResult { return builder.getI64IntegerAttr(v); });
  SmallVector<OpFoldResult> dest_size_values = llvm::map_to_vector<4>(
      dest_sizes,
      [&](int64_t v) -> OpFoldResult { return builder.getI64IntegerAttr(v); });

  // Delegate to the OpFoldResult builder
  build(builder, result, source, source_indices, source_size_values,
        destination, dest_indices, dest_size_values);
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(::mlir::OpBuilder& builder,
                       ::mlir::OperationState& state,
                       ::mlir::ValueRange lower_bounds,
                       ::mlir::ValueRange upper_bounds,
                       ::mlir::ValueRange steps,
                       int64_t num_instances) {
  state.addOperands(lower_bounds);
  state.addOperands(upper_bounds);
  state.addOperands(steps);
  state.addAttribute("operandSegmentSizes",
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(lower_bounds.size()),
                         static_cast<int32_t>(upper_bounds.size()),
                         static_cast<int32_t>(steps.size())}));
  state.addAttribute("num_instances", builder.getI64IntegerAttr(num_instances));

  // Create the body region with a single block. The block has one index
  // argument per induction variable plus one trailing instance-id argument.
  Region* body = state.addRegion();
  Block& block = body->emplaceBlock();
  Type index_ty = builder.getIndexType();
  unsigned num_loops = lower_bounds.size();
  SmallVector<Type> arg_types(num_loops + 1, index_ty);
  SmallVector<Location> arg_locs(num_loops + 1, state.location);
  block.addArguments(arg_types, arg_locs);
}

ParseResult ParallelOp::parse(OpAsmParser& parser, OperationState& result) {
  auto& builder = parser.getBuilder();
  Type index_ty = builder.getIndexType();

  // Parse the block-argument list: `(` %iv1, %iv2, ..., %inst `)`.
  SmallVector<OpAsmParser::Argument> region_args;
  if (parser.parseLParen()) {
    return failure();
  }
  if (parser.parseCommaSeparatedList([&]() -> ParseResult {
        OpAsmParser::Argument arg;
        if (parser.parseArgument(arg)) {
          return failure();
        }
        arg.type = index_ty;
        region_args.push_back(arg);
        return success();
      })) {
    return failure();
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (region_args.size() < 2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected at least one induction variable plus an "
                            "instance id (got ")
           << region_args.size() << " block arguments)";
  }
  unsigned num_loops = region_args.size() - 1;

  // Parse `=` `(` lower_bounds `)`.
  SmallVector<OpAsmParser::UnresolvedOperand> lbs;
  if (parser.parseEqual() || parser.parseLParen() ||
      parser.parseOperandList(lbs, num_loops) || parser.parseRParen()) {
    return failure();
  }

  // Parse `to` `(` upper_bounds `)`.
  SmallVector<OpAsmParser::UnresolvedOperand> ubs;
  if (parser.parseKeyword("to") || parser.parseLParen() ||
      parser.parseOperandList(ubs, num_loops) || parser.parseRParen()) {
    return failure();
  }

  // Parse `step` `(` steps `)`.
  SmallVector<OpAsmParser::UnresolvedOperand> steps;
  if (parser.parseKeyword("step") || parser.parseLParen() ||
      parser.parseOperandList(steps, num_loops) || parser.parseRParen()) {
    return failure();
  }

  // Parse `distribute` `(` `num_instances` `=` integer-literal `)`.
  if (parser.parseKeyword("distribute") || parser.parseLParen() ||
      parser.parseKeyword("num_instances") || parser.parseEqual()) {
    return failure();
  }
  int64_t num_instances = 0;
  if (parser.parseInteger(num_instances)) {
    return failure();
  }
  if (parser.parseRParen()) {
    return failure();
  }
  result.addAttribute("num_instances",
                      builder.getI64IntegerAttr(num_instances));

  // Resolve operands.
  if (parser.resolveOperands(lbs, index_ty, result.operands) ||
      parser.resolveOperands(ubs, index_ty, result.operands) ||
      parser.resolveOperands(steps, index_ty, result.operands)) {
    return failure();
  }

  result.addAttribute("operandSegmentSizes",
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(lbs.size()),
                          static_cast<int32_t>(ubs.size()),
                          static_cast<int32_t>(steps.size())}));

  // Parse the region using the named block arguments.
  Region* body = result.addRegion();
  if (parser.parseRegion(*body, region_args)) {
    return failure();
  }

  // Optional trailing attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void ParallelOp::print(OpAsmPrinter& p) {
  // Print block-argument list: `(` %iv1, ..., %inst `)`. Block args are all
  // index-typed by construction (see verify), so we only print the SSA names
  // here — the parser binds them as index without type annotations, matching
  // the spec's surface form.
  p << " (";
  llvm::interleaveComma(getBody().front().getArguments(), p,
                        [&](BlockArgument arg) { p.printOperand(arg); });
  p << ")";

  // Print `= (lbs) to (ubs) step (steps)`.
  p << " = (";
  p.printOperands(getLowerBounds());
  p << ") to (";
  p.printOperands(getUpperBounds());
  p << ") step (";
  p.printOperands(getSteps());
  p << ")";

  // Print `distribute(num_instances = N)`.
  p << " distribute(num_instances = " << getNumInstances() << ")";

  // Print the region.
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  // Print remaining attributes (suppress the ones already printed inline).
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes",
                                           "num_instances"});
}

LogicalResult ParallelOp::verify() {
  unsigned num_lbs = getLowerBounds().size();
  unsigned num_ubs = getUpperBounds().size();
  unsigned num_steps = getSteps().size();

  if (num_lbs != num_ubs || num_lbs != num_steps) {
    return emitOpError(
               "expected matching numbers of lower bounds, upper bounds, and "
               "steps; got ")
           << num_lbs << ", " << num_ubs << ", and " << num_steps;
  }

  if (num_lbs < 1) {
    return emitOpError("expected at least one induction variable");
  }

  if (getNumInstances() < 1) {
    return emitOpError("expected num_instances >= 1, got ") << getNumInstances();
  }

  Block& block = getBody().front();
  unsigned expected_args = num_lbs + 1;
  if (block.getNumArguments() != expected_args) {
    return emitOpError("expected ")
           << expected_args
           << " block arguments (one per induction variable plus the instance "
              "id), got "
           << block.getNumArguments();
  }
  for (BlockArgument arg : block.getArguments()) {
    if (!arg.getType().isIndex()) {
      return emitOpError(
          "all block arguments (induction variables and instance id) must be "
          "of type index");
    }
  }

  if (block.empty() || !isa<ParallelYieldOp>(block.back())) {
    return emitOpError(
        "region must be terminated by a ktdf.parallel_yield operation");
  }

  return success();
}
