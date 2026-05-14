add_library(SchedulerUtils
  OperationUtils.cpp
  SchedulerExtContext.cpp
)

target_link_libraries(SchedulerUtils
  PUBLIC
  MLIRIR
  KTDFOps
)

target_include_directories(SchedulerUtils
  PUBLIC
  ${CMAKE_SOURCE_DIR}
)