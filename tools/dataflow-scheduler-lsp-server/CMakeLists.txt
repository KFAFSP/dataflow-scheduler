add_dataflow_scheduler_tool(dataflow-scheduler-lsp-server
  dataflow-scheduler-lsp-server.cpp

  EXPORT_NAME lsp-server
)
target_link_libraries(dataflow-scheduler-lsp-server PRIVATE 
  DataflowScheduler::RegisterEverything
)

mlir_target_link_libraries(dataflow-scheduler-lsp-server PRIVATE 
  MLIRLspServerLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
)
llvm_update_compile_flags(dataflow-scheduler-lsp-server)
