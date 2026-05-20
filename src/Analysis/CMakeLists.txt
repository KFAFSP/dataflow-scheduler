add_ktir_dialect_library(KTIRSchedulerAnalysis
  OperationTree.cpp
  PipelineTree.cpp
  StageGrouping.cpp

  LINK_LIBS PUBLIC
    KTIRKTDFDialect
)

# FIXME: Clearly, MLIR libs are not being linked.
