add_library(SchedulerUtils
  OperationUtils.cpp
)

target_link_libraries(SchedulerUtils
  PUBLIC
  MLIRIR
)

target_include_directories(SchedulerUtils
  PUBLIC
  ${CMAKE_SOURCE_DIR}
)