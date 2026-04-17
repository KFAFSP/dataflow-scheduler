cmake_minimum_required(VERSION 3.20.0)
project(ktir-scheduler)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SCHEDULER_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SCHEDULER_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(SCHEDULER_TOOLS_DIR "${CMAKE_BINARY_DIR}/scheduler/bin")

# Add cmake directory to search for custom cmake functions
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
message(STATUS "cmake_module_path ${CMAKE_MODULE_PATH}")

include(MLIR)
include(clang-format)
# for lit testings
set(LLVM_EXTERNAL_LIT "${MLIR_LLVM_SOURCE_DIR}/utils/lit/lit.py" CACHE STRING "Command used to spawn lit")

# Add KTIR dependency
set(KTIR_PROJ_SRC $ENV{KTIR_PROJ_SRC} CACHE STRING "Points to KTIR source directory")
if("${KTIR_PROJ_SRC}" STREQUAL "")
  message(FATAL_ERROR "Environment variable KTIR_PROJ_SRC was not set")
endif()
message(STATUS "KTIR_PROJ_SRC:${KTIR_PROJ_SRC}")

set(KTIR_PROJ_BUILD $ENV{KTIR_PROJ_BUILD} CACHE STRING "Points to KTIR build directory")
if("${KTIR_PROJ_BUILD}" STREQUAL "")
  message(FATAL_ERROR "Environment variable KTIR_PROJ_BUILD was not set")
endif()
message(STATUS "KTIR_PROJ_BUILD:${KTIR_PROJ_BUILD}")

# Find KTIR libraries and includes
find_library(KTDP_LIB MLIRKtdp PATHS ${KTIR_PROJ_BUILD}/lib NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found KTDP library: ${KTDP_LIB}")

include_directories(${SCHEDULER_SRC_ROOT})
include_directories(${SCHEDULER_SRC_ROOT}/src)
include_directories(${SCHEDULER_BIN_ROOT})
include_directories(${KTIR_PROJ_SRC}/include)
include_directories(${KTIR_PROJ_BUILD}/include)

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(test)

message(STATUS "Finished loading KTIR-Scheduler CMake configuration")