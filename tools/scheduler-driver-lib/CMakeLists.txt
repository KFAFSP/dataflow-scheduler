add_library(SchedulerDriverLib
    scheduler-main.cpp
)

target_link_libraries(SchedulerDriverLib
    PUBLIC
    SchedulerInitAllDialects
    MLIR
    LLVM
)