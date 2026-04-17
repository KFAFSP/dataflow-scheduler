find_program(CLANG_FORMAT NAMES "clang-format" PATHS ${LLVM_PROJ_BUILD}/bin NO_DEFAULT_PATH)
if(NOT CLANG_FORMAT)
    message(FATAL_ERROR "clang-format not found in  ${LLVM_PROJ_BUILD}/bin Enable clang during llvm build")
else()
    message(STATUS "Found clang-format at ${CLANG_FORMAT}")
endif()

file(GLOB_RECURSE SCHEDULER_SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
                                         ${PROJECT_SOURCE_DIR}/src/*.h
                                         ${PROJECT_SOURCE_DIR}/tools/*.h
                                         ${PROJECT_SOURCE_DIR}/tools/*.cpp)

add_custom_target(
        format-scheduler
        COMMAND ${CLANG_FORMAT}
        -style=file
        -i
        ${SCHEDULER_SOURCE_FILES})

add_custom_target(check-scheduler-format
        COMMENT "Checking clang-format"
        COMMAND ${CLANG_FORMAT} -style=file
                 --output-replacements-xml
                 ${SCHEDULER_SOURCE_FILES}
                 | tee ${CMAKE_BINARY_DIR}/check_format_scheduler_file.txt | grep -c "replacement " |
                 tr -d "[:cntrl:]" && echo " replacements necessary"
            # WARNING: fix to stop with error if there are problems
            COMMAND ! grep -c "replacement "
            ${CMAKE_BINARY_DIR}/check_format_scheduler_file.txt > /dev/null
            COMMENT "Checking scheduler format compliance"
)
