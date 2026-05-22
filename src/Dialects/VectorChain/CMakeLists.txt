add_ktir_dialect(VectorChainOps vectorchain)
add_ktir_doc(VectorChainOps VectorChainDialect Dialects/ -gen-dialect-doc -dialect=vectorchain)

set(LLVM_TARGET_DEFINITIONS VectorChainEnums.td)
mlir_tablegen(VectorChainEnums.hpp.inc -gen-enum-decls)
mlir_tablegen(VectorChainEnums.cpp.inc -gen-enum-defs)
add_ktir_tablegen_target(KTIRVectorChainEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS VectorChainOps.td)
mlir_tablegen(VectorChainAttributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=vectorchain)
mlir_tablegen(VectorChainAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=vectorchain)
add_ktir_tablegen_target(KTIRVectorChainAttributesIncGen)

add_ktir_dialect_library(KTIRVectorChainDialect
    VectorChainOps.cpp
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
    KTIRVectorChainOpsIncGen
    KTIRVectorChainEnumsIncGen
    KTIRVectorChainAttributesIncGen

  LINK_LIBS PUBLIC
    KTIRDataflowDialect
)

mlir_target_link_libraries(KTIRVectorChainDialect PUBLIC
  MLIRIR
)
