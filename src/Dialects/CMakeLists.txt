add_subdirectory(KTDF)

add_library(SchedulerInitMLIRDialects init-mlir-dialects.cpp)
add_library(SchedulerInitKTIRDialects init-ktir-dialects.cpp)
add_library(SchedulerInitSchedulerDialects init-scheduler-dialects.cpp)

target_link_libraries(SchedulerInitKTIRDialects PUBLIC ${KTDP_LIB})
target_link_libraries(SchedulerInitSchedulerDialects PUBLIC KTDFOps)

add_library(SchedulerInitAllDialects init-all-dialects.cpp)
target_link_libraries(SchedulerInitAllDialects
  PUBLIC
  SchedulerInitMLIRDialects
  SchedulerInitKTIRDialects
  SchedulerInitSchedulerDialects
)

add_dependencies(SchedulerInitSchedulerDialects KTDFOps)