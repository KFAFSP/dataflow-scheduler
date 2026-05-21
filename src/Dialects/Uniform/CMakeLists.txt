add_ktir_dialect(UniformOps uniform)
add_ktir_doc(UniformOps UniformDialect Dialects/ -gen-dialect-doc -dialect=uniform)

add_ktir_dialect_library(KTIRUniformDialect
    UniformOps.cpp
  
  ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
    KTIRUniformOpsIncGen

  LINK_LIBS PRIVATE
    KTIRDataflowDialect
)

mlir_target_link_libraries(KTIRUniformDialect PUBLIC
  MLIRIR
  # TODO: Add dependencies.
)
