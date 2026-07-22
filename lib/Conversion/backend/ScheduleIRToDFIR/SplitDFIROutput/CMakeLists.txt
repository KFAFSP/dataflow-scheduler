add_dataflow_scheduler_library(DataflowSchedulerSplitDFIROutput
    SplitDFIROutput.cpp

  GROUP CONVERSION
  EXPORT_NAME SplitDFIROutput

  DEPENDS
    DataflowSchedulerScheduleIRToDFIRPassesIncGen
    
  LINK_LIBS PUBLIC  
    DataflowSchedulerDialects::DataflowDialect

  LINK_MLIR_LIBS
    MLIRFuncDialect
    MLIRIR
    MLIRPass
)
