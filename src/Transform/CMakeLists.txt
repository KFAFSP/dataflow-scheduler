set(LLVM_TARGET_DEFINITIONS Passes.td)
scheduler_tablegen(Passes.h.inc -gen-pass-decls -name SchedulerTransforms)
add_public_tablegen_target(SchedulerTransformsPassIncGen)

add_library(SchedulerTransforms
    DummyTransform.cpp
)

target_link_libraries(SchedulerTransforms
    PUBLIC
    MLIRIR
    MLIRPass
)

add_dependencies(SchedulerTransforms SchedulerTransformsPassIncGen)