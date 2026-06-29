set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name KTDFTransforms)
add_dataflow_scheduler_tablegen_target(DataflowSchedulerKTDFTransformsPassesIncGen)

add_dataflow_scheduler_doc(Passes KTDFPasses ./ -gen-pass-doc)
