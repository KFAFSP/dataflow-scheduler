set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ScheduleIRToDFIR)
add_dataflow_scheduler_tablegen_target(DataflowSchedulerScheduleIRToDFIRPassesIncGen)

add_dataflow_scheduler_doc(Passes SchedulerScheduleIRToDFIRPasses ./ -gen-pass-doc)
