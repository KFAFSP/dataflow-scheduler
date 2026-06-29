# NOTE: These are for tablegen only!
include_directories(SYSTEM 
  ${KTIR_INCLUDE_DIRS}
  ${DataflowSchedulerDialects_INCLUDE_DIRS}
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(dataflow-scheduler)
