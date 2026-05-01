set(LLVM_TARGET_DEFINITIONS Passes.td)
scheduler_tablegen(Passes.h.inc -gen-pass-decls -name SchedulerTransforms)
add_public_tablegen_target(SchedulerTransformsPassIncGen)

add_library(SchedulerTransforms
    ComputeGroupExtraction.cpp
    TileSCFForLoops.cpp
    StripMineSCFForLoops.cpp
    PrintPipelineTree.cpp
    StageCoarsening.cpp
    StructuralLegalizer.cpp
)

target_link_libraries(SchedulerTransforms
    PUBLIC
    KTDFOps
    SchedulerUtils
    SchedulerAnalysis
)

add_dependencies(SchedulerTransforms SchedulerTransformsPassIncGen)
