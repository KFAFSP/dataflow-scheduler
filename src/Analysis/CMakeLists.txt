add_library(SchedulerAnalysis
    OperationTree.cpp
    PipelineTree.cpp
    StageGrouping.cpp
)

add_dependencies(SchedulerAnalysis KTDFOps)
