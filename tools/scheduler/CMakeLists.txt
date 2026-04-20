add_executable(scheduler
    scheduler.cpp
    scheduler-pipeline.cpp
)

set(LIBS
    SchedulerDriverLib
    SchedulerInitAllDialects
    SchedulerTransforms
)

target_link_libraries(scheduler PRIVATE ${LIBS} MLIR LLVM)

set_target_properties(scheduler
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SCHEDULER_TOOLS_DIR}"
)