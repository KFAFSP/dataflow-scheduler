add_subdirectory(Agen)
add_subdirectory(Dataflow)
add_subdirectory(KTDF)
add_subdirectory(Uniform)
add_subdirectory(VectorChain)

add_ktir_library(KTIRSchedulerInitMLIRDialects 
    init-mlir-dialects.cpp

  PARTIAL_SOURCES_INTENDED
)

mlir_target_link_libraries(KTIRSchedulerInitMLIRDialects PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
)

add_ktir_library(KTIRSchedulerInitKTIRDialects 
    init-ktir-dialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRKtdpDialect
)

add_ktir_library(KTIRSchedulerInitSchedulerDialects 
    init-scheduler-dialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRKTDFDialect
    KTIRAgenDialect
    KTIRDataflowDialect
    KTIRUniformDialect
    KTIRVectorChainDialect
    KTIRSchedulerTransforms
)

add_ktir_library(KTIRSchedulerInitAllDialects 
    init-all-dialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRSchedulerInitMLIRDialects
    KTIRSchedulerInitKTIRDialects
    KTIRSchedulerInitSchedulerDialects
)
