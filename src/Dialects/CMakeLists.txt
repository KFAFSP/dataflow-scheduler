add_subdirectory(KTDF)

add_ktir_library(KTIRSchedulerInitMLIRDialects 
    init-mlir-dialects.cpp

  PARTIAL_SOURCES_INTENDED
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
)

add_ktir_library(KTIRSchedulerInitAllDialects 
    init-all-dialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    KTIRSchedulerInitMLIRDialects
    KTIRSchedulerInitKTIRDialects
    KTIRSchedulerInitSchedulerDialects
)

# FIXME: Clearly, MLIR libs are not being linked.
