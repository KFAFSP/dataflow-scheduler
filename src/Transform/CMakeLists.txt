set(LLVM_TARGET_DEFINITIONS Passes.td)
scheduler_tablegen(Passes.h.inc -gen-pass-decls -name SchedulerTransforms)
add_public_tablegen_target(SchedulerTransformsPassIncGen)

add_library(SchedulerTransforms
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    PipelineTree/Legalizer.cpp
    PipelineTree/Materializer.cpp
    StageCoarsening/BufferExpansion.cpp
    StageCoarsening/ScopeCorrection.cpp
    StageCoarsening.cpp
    StripMineSCFForLoops.cpp
    TileSCFForLoops.cpp
    Utils/CustomTiling.cpp
)

target_link_libraries(SchedulerTransforms
    PUBLIC
    KTDFOps
    SchedulerUtils
    SchedulerAnalysis
)

add_dependencies(SchedulerTransforms SchedulerTransformsPassIncGen)
