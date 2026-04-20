add_executable(scheduler
    scheduler.cpp
    scheduler-pipeline.cpp
)

target_link_libraries(scheduler
    PRIVATE
    SchedulerDriverLib
    SchedulerTransforms
)

set_target_properties(scheduler
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SCHEDULER_TOOLS_DIR}"
)