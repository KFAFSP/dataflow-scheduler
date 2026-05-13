set(LLVM_TARGET_DEFINITIONS Passes.td)
scheduler_tablegen(Passes.h.inc -gen-pass-decls -name SchedulerTransforms)
add_public_tablegen_target(SchedulerTransformsPassIncGen)

add_library(SchedulerTransforms
    BroadcastPromotion.cpp
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    PipelineTree/Legalizer.cpp
    PipelineTree/Materializer.cpp
    ReusePromotion/Common/InvarianceCheck.cpp
    ReusePromotion/Common/PipelineScope.cpp
    StageCoarsening/BufferExpansion.cpp
    StageCoarsening/ScopeCorrection.cpp
    StageCoarsening.cpp
    StripMineSCFForLoops.cpp
    TileSCFForLoops.cpp
    Utils/CustomLinalgTiling.cpp
    Utils/Utils.cpp
)

target_link_libraries(SchedulerTransforms
    PUBLIC
    KTDFOps
    SchedulerUtils
    SchedulerAnalysis
)

add_dependencies(SchedulerTransforms SchedulerTransformsPassIncGen)
