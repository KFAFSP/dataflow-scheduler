set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SchedulerTransforms)
add_dataflow_scheduler_tablegen_target(DataflowSchedulerTransformsPassesIncGen)

add_dataflow_scheduler_doc(Passes SchedulerPasses ./ -gen-pass-doc)
