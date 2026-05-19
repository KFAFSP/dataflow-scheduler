set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SchedulerTransforms)
add_ktir_tablegen_target(KTIRSchedulerTransformsPassIncGen)

add_ktir_doc(Passes KTDFPasses ./ -gen-pass-doc)

add_ktir_dialect_library(KTIRSchedulerTransforms
    BroadcastPromotion.cpp
    Common/ApplicableUnits.cpp
    Common/PipelineScope.cpp
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    DoubleBuffering.cpp
    ParallelizeLoopsAcrossInstances.cpp
    PathExpansion.cpp
    PathExpansion/Materializer.cpp
    PathExpansion/Planner.cpp
    PipelineTree/Legalizer.cpp
    ReusePromotion/Common/CandidateSearch.cpp
    ReusePromotion/Common/InvarianceCheck.cpp
    ReusePromotion/Common/RegionClonePrune.cpp
    ReusePromotion/Common/WriteSetScan.cpp
    StageCoarsening/BufferExpansion.cpp
    StageCoarsening/Materializer.cpp
    StageCoarsening/ScopeCorrection.cpp
    StageCoarsening.cpp
    StripMineSCFForLoops.cpp
    TileSCFForLoops.cpp
    Utils/CustomLinalgTiling.cpp
    Utils/Utils.cpp

    PARTIAL_SOURCES_INTENDED
  
  DEPENDS
    KTIRSchedulerTransformsPassIncGen

  LINK_LIBS PUBLIC
    KTIRKTDFDialect
    KTIRKtdpDialect
    KTIRSchedulerUtils
    KTIRSchedulerAnalysis
)

mlir_target_link_libraries(KTIRSchedulerTransforms PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRSCFUtils
  MLIRMathDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
)
