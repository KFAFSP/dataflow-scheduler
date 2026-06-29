add_dataflow_scheduler_library(DataflowSchedulerArchViews
    MemoryTree.cpp
    ResourceKinds.cpp
    RoutingGraph.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Analysis/ArchViews

  EXPORT_NAME ArchViews

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFArchDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    KTIR::KtdpDialect
)

mlir_target_link_libraries(DataflowSchedulerArchViews PUBLIC
  MLIRIR
)