add_executable(scheduler
    scheduler.cpp
    scheduler-pipeline.cpp
    ../../src/Transform/DummyTransform.cpp
)

target_link_libraries(scheduler PRIVATE SchedulerDriverLib MLIR LLVM)

set_target_properties(scheduler
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SCHEDULER_TOOLS_DIR}"
)