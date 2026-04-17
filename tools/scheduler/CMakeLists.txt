add_executable(scheduler scheduler.cpp)

set(LIBS
    SchedulerDriverLib
    SchedulerInitAllDialects
)

target_link_libraries(scheduler PRIVATE ${LIBS} MLIR LLVM)

set_target_properties(scheduler
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SCHEDULER_TOOLS_DIR}"
)