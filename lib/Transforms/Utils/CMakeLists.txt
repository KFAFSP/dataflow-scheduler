add_dataflow_scheduler_library(DataflowSchedulerTransformUtils
    CustomLinalgTiling.cpp
    Hoisting.cpp
    PipelineTreeLegalizer.cpp
    RegionClonePrune.cpp
    SCFTilingUtils.cpp
    Utils.cpp

  EXPORT_NAME TransformUtils

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFDialect

  LINK_MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRArithDialect
    MLIRIR
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRSideEffectInterfaces
)
