add_ktir_dialect(DataflowOps dataflow)
add_ktir_doc(DataflowOps DataflowDialect Dialects/ -gen-dialect-doc -dialect=dataflow)

set(LLVM_TARGET_DEFINITIONS DataflowEnums.td)
mlir_tablegen(DataflowEnums.hpp.inc -gen-enum-decls)
mlir_tablegen(DataflowEnums.cpp.inc -gen-enum-defs)
add_ktir_tablegen_target(KTIRDataflowEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS DataflowOps.td)
mlir_tablegen(DataflowAttributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=dataflow)
mlir_tablegen(DataflowAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=dataflow)
add_ktir_tablegen_target(KTIRDataflowAttributesIncGen)

add_ktir_dialect_library(KTIRDataflowDialect
    DataflowOps.cpp
    DataflowTypes.cpp
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
    KTIRDataflowOpsIncGen
    KTIRDataflowEnumsIncGen
    KTIRDataflowAttributesIncGen
)

mlir_target_link_libraries(KTIRDataflowDialect PUBLIC
  MLIRIR
  MLIRSCFDialect
)
