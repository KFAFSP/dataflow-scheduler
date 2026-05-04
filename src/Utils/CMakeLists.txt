add_library(SchedulerUtils
  OperationUtils.cpp
  SchedulerExtContext.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

target_link_libraries(SchedulerUtils
  PUBLIC
  MLIRIR
)

target_include_directories(SchedulerUtils
  PUBLIC
  ${CMAKE_SOURCE_DIR}
)