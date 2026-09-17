add_dataflow_scheduler_library(DataflowSchedulerDataflowTransforms
    MappableExtensions.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/Dataflow/Transforms

  GROUP DIALECT
  EXPORT_NAME DataflowTransforms

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
)
