set(LLVM_TARGET_DEFINITIONS KTDFOps.td)
scheduler_tablegen(KTDFOpsDialect.hpp.inc -gen-dialect-decls -dialect=ktdf)
scheduler_tablegen(KTDFOpsDialect.cpp.inc -gen-dialect-defs -dialect=ktdf)
scheduler_tablegen(KTDFOps.hpp.inc -gen-op-decls)
scheduler_tablegen(KTDFOps.cpp.inc -gen-op-defs)
scheduler_tablegen(KTDFOpsEnums.hpp.inc -gen-enum-decls)
scheduler_tablegen(KTDFOpsEnums.cpp.inc -gen-enum-defs)
scheduler_tablegen(KTDFOpsAttributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=ktdf)
scheduler_tablegen(KTDFOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ktdf)
scheduler_tablegen(KTDFOps.md -gen-dialect-doc -dialect=ktdf)
add_public_tablegen_target(KTDFOpsIncGen)

set(LLVM_TARGET_DEFINITIONS KTDFTypes.td)
scheduler_tablegen(KTDFTypes.hpp.inc -gen-typedef-decls -typedefs-dialect=ktdf)
scheduler_tablegen(KTDFTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ktdf)
add_public_tablegen_target(KTDFTypesIncGen)

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

add_dependencies(KTDFOps KTDFOpsIncGen KTDFTypesIncGen)