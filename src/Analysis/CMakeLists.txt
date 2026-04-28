add_library(SchedulerAnalysis
    PipelineTree.cpp
    StageGrouping.cpp
)

add_dependencies(SchedulerAnalysis KTDFOps)
