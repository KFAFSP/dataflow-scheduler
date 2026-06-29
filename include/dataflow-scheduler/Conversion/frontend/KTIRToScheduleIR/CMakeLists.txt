set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name KTIRToScheduleIR)
add_dataflow_scheduler_tablegen_target(DataflowSchedulerKTIRToScheduleIRPassesIncGen)

add_dataflow_scheduler_doc(Passes SchedulerKTIRToScheduleIRPasses ./ -gen-pass-doc)
