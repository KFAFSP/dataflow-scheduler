add_ktir_dialect_library(KTIRSchedulerAnalysis
  OperationTree.cpp
  PipelineTree.cpp
  StageGrouping.cpp

  LINK_LIBS PUBLIC
    KTIRKTDFDialect
    KTIRSchedulerUtils
)

mlir_target_link_libraries(KTIRSchedulerAnalysis PRIVATE
  MLIRSCFDialect
)