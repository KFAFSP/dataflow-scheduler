add_mlir_public_c_api_library(KTIRAgenCAPI
    Agen.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRAgenDialect
)

add_mlir_public_c_api_library(KTIRDataflowCAPI
    Dataflow.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRDataflowDialect
)

add_mlir_public_c_api_library(KTIRKTDFCAPI
    KTDF.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRKTDFDialect
)

add_mlir_public_c_api_library(KTIRRegisterEverythingCAPI
    RegisterEverything.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRSchedulerInitAllDialects
)

add_mlir_public_c_api_library(KTIRUniformCAPI
    Uniform.cpp
    
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRUniformDialect
)

add_mlir_public_c_api_library(KTIRVectorChainCAPI
    VectorChain.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRVectorChainDialect
)
