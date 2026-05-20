add_ktir_library(KTIRSchedulerUtils
    OperationUtils.cpp
    SchedulerExtContext.cpp
  LINK_LIBS PRIVATE
    KTIRKTDFDialect
)

mlir_target_link_libraries(KTIRSchedulerUtils PUBLIC
  MLIRIR
)
