add_ktir_dialect(KTDFOps ktdf)
add_ktir_doc(KTDFOps KTDFDialect Dialects/ -gen-dialect-doc -dialect=ktdf)

set(LLVM_TARGET_DEFINITIONS KTDFAttributes.td)
mlir_tablegen(KTDFAttributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=ktdf)
mlir_tablegen(KTDFAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ktdf)
add_ktir_tablegen_target(KTIRKTDFAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS KTDFEnums.td)
mlir_tablegen(KTDFEnums.hpp.inc -gen-enum-decls)
mlir_tablegen(KTDFEnums.cpp.inc -gen-enum-defs)
add_ktir_tablegen_target(KTIRKTDFEnumsIncGen)

add_ktir_dialect_library(KTIRKTDFDialect
    KTDFOps.cpp
    KTDFOps.hpp
    KTDFTypes.cpp
    KTDFTypes.hpp
  
  ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
    KTIRKTDFOpsIncGen
    KTIRKTDFAttributesIncGen
    KTIRKTDFEnumsIncGen
)

mlir_target_link_libraries(KTIRKTDFDialect PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRControlFlowInterfaces
  MLIRDialectUtils
  MLIRSCFDialect
)
