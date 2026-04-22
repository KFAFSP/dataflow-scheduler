set(LLVM_TARGET_DEFINITIONS Passes.td)
scheduler_tablegen(Passes.h.inc -gen-pass-decls -name SchedulerTransforms)
add_public_tablegen_target(SchedulerTransformsPassIncGen)

add_library(SchedulerTransforms
    DummyTransform.cpp
)

add_dependencies(SchedulerTransforms SchedulerTransformsPassIncGen)