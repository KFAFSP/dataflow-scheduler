add_dataflow_scheduler_library(DataflowSchedulerSplitDFIROutput
    SplitDFIROutput.cpp

  GROUP CONVERSION
  EXPORT_NAME SplitDFIROutput

  DEPENDS
    DataflowSchedulerScheduleIRToDFIRPassesIncGen
    
  LINK_LIBS PUBLIC  
    DataflowSchedulerDialects::DataflowDialect
)

mlir_target_link_libraries(DataflowSchedulerSplitDFIROutput PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
