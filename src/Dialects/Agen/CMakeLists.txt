add_ktir_dialect(AgenOps agen)
add_ktir_doc(AgenOps AgenDialect Dialects/ -gen-dialect-doc -dialect=agen)

set(LLVM_TARGET_DEFINITIONS AgenOps.td)
mlir_tablegen(AgenAttributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=agen)
mlir_tablegen(AgenAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=agen)
add_ktir_tablegen_target(KTIRAgenAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS AgenEnums.td)
mlir_tablegen(AgenEnums.hpp.inc -gen-enum-decls)
mlir_tablegen(AgenEnums.cpp.inc -gen-enum-defs)
add_ktir_tablegen_target(KTIRAgenEnumsIncGen)

add_ktir_dialect_library(KTIRAgenDialect
    AgenOps.cpp
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
    KTIRAgenOpsIncGen
    KTIRAgenAttributesIncGen
    KTIRAgenEnumsIncGen

  LINK_LIBS PUBLIC
    KTIRDataflowDialect
    KTIRVectorChainDialect
)

mlir_target_link_libraries(KTIRAgenDialect PUBLIC
  MLIRIR
  # These are used by Utils.cpp, but do not require dialect loading.
  MLIRAffineDialect
  MLIRAffineAnalysis
)
