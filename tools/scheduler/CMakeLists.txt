add_executable(scheduler
    scheduler.cpp
    scheduler-pipeline.cpp
    ../../src/Transform/DummyTransform.cpp
)

# FIXME: this is a temporary fix until we resolve the dialect registration issue preventing us from building transforms as a separate lib.
add_dependencies(scheduler SchedulerTransformsPassIncGen)

target_link_libraries(scheduler PRIVATE SchedulerDriverLib MLIR LLVM)

set_target_properties(scheduler
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SCHEDULER_TOOLS_DIR}"
)