add_dataflow_scheduler_library(DataflowSchedulerUtils
    SchedulerExtContext.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Utils

  EXPORT_NAME Utils

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
)

mlir_target_link_libraries(DataflowSchedulerUtils PUBLIC
  MLIRIR
)
