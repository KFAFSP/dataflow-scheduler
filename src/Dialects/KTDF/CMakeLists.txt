set(LLVM_TARGET_DEFINITIONS KTDFEnums.td)
scheduler_tablegen(KTDFEnums.hpp.inc -gen-enum-decls)
scheduler_tablegen(KTDFEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(KTDFEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS KTDFOps.td)
scheduler_tablegen(KTDFOpsDialect.hpp.inc -gen-dialect-decls -dialect=ktdf)
scheduler_tablegen(KTDFOpsDialect.cpp.inc -gen-dialect-defs -dialect=ktdf)
scheduler_tablegen(KTDFOps.hpp.inc -gen-op-decls)
scheduler_tablegen(KTDFOps.cpp.inc -gen-op-defs)
scheduler_tablegen(KTDFOps.md -gen-dialect-doc -dialect=ktdf)
add_public_tablegen_target(KTDFOpsIncGen)

set(LLVM_TARGET_DEFINITIONS KTDFTypes.td)
scheduler_tablegen(KTDFTypes.hpp.inc -gen-typedef-decls -typedefs-dialect=ktdf)
scheduler_tablegen(KTDFTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ktdf)
add_public_tablegen_target(KTDFTypesIncGen)

set(LLVM_TARGET_DEFINITIONS KTDFAttributes.td)
scheduler_tablegen(KTDFTypesAttributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=ktdf)
scheduler_tablegen(KTDFTypesAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ktdf)
add_public_tablegen_target(KTDFAttributesIncGen)

add_library(KTDFOps
        KTDFOps.cpp
        KTDFOps.hpp
        KTDFTypes.cpp
        KTDFTypes.hpp)

target_include_directories(KTDFOps
        PUBLIC
        ${LLVM_PROJ_SRC}
        ${PROJECT_BINARY_DIR}
)

add_dependencies(KTDFOps KTDFEnumsIncGen KTDFOpsIncGen KTDFTypesIncGen KTDFAttributesIncGen)
