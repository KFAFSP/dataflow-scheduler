#get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
#get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(SCHEDULER_MLIR_LIBS
#    ${dialect_libs}
#    ${conversion_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
)

add_executable(scheduler
    scheduler.cpp
    scheduler-pipeline.cpp
)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    target_link_libraries(scheduler
        PRIVATE
        -Wl,--whole-archive
        ${SCHEDULER_MLIR_LIBS}
        -Wl,--no-whole-archive
        SchedulerInitAllDialects
        SchedulerDriverLib
        SchedulerTransforms
    )
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    foreach(mlir_lib IN LISTS SCHEDULER_MLIR_LIBS)
        target_link_libraries(scheduler PRIVATE -Wl,-force_load ${mlir_lib})
    endforeach()
    target_link_libraries(scheduler
        PRIVATE
        SchedulerInitAllDialects
        SchedulerDriverLib
        SchedulerTransforms
    )
else()
    message(WARNING "Untested system '${CMAKE_SYSTEM_NAME}': linking scheduler without whole-archive/all_load for MLIR libs")
    target_link_libraries(scheduler
        PRIVATE
        SchedulerInitAllDialects
        SchedulerDriverLib
        SchedulerTransforms
        ${SCHEDULER_MLIR_LIBS}
    )
endif()

set_target_properties(scheduler
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${SCHEDULER_TOOLS_DIR}"
)
