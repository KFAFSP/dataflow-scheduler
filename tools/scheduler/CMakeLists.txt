# FIXME: Consider scheduler executable packaging and then address linking.

add_ktir_tool(scheduler
  scheduler.cpp
  scheduler-pipeline.cpp
)
target_link_libraries(scheduler PRIVATE
  KTIRSchedulerInitAllDialects
  KTIRSchedulerDriverLib
  KTIRSchedulerTransforms
)

llvm_update_compile_flags(scheduler)
mlir_target_link_libraries(scheduler PRIVATE
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
)
