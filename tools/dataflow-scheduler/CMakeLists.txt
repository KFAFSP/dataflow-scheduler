add_dataflow_scheduler_tool(dataflow-scheduler
    dataflow-scheduler-main.cpp
    dataflow-scheduler.cpp

  EXPORT_NAME dataflow-scheduler
)
target_link_libraries(dataflow-scheduler PRIVATE
  # NOTE: This is _not_ part of libMLIR.so!
  MLIRMlirOptMain

  DataflowScheduler::Pipeline
)

mlir_target_link_libraries(dataflow-scheduler PRIVATE
  MLIROptLib
)
llvm_update_compile_flags(dataflow-scheduler)
