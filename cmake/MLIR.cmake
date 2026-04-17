if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD) # for llvm tablegen
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_HAS_NO_ASAN")
set(LLVM_PROJ_SRC $ENV{LLVM_PROJ_SRC} CACHE STRING "Points to llvm-project source directory")
set(LLVM_PROJ_BUILD $ENV{LLVM_PROJ_BUILD} CACHE STRING "Points to llvm-project's build directory")

if("${LLVM_PROJ_SRC}" STREQUAL "")
  message(FATAL_ERROR "Environment variable LLVM_PROJ_SRC was not set, use local")
endif()
if("${LLVM_PROJ_BUILD}" STREQUAL "")
  message(WARNING "Environment variable LLVM_PROJ_BUILD was not set, use default")
  set(LLVM_PROJ_BUILD "${LLVM_PROJ_SRC}/build" CACHE STRING "Points to llvm-project's build directory" FORCE)
endif()
string(REGEX REPLACE "/$" "" LLVM_PROJ_SRC ${LLVM_PROJ_SRC})
string(REGEX REPLACE "/$" "" LLVM_PROJ_BUILD ${LLVM_PROJ_BUILD})
message(STATUS "LLVM_PROJ_SRC:${LLVM_PROJ_SRC}")
message(STATUS "LLVM_PROJ_BUILD:${LLVM_PROJ_BUILD}")

set(MLIR_LLVM_SOURCE_DIR ${LLVM_PROJ_SRC}/llvm)
set(MLIR_SOURCE_DIR ${LLVM_PROJ_SRC}/mlir)
set(MLIR_LLVM_BUILD_DIR ${LLVM_PROJ_BUILD})
set(MLIR_LLVM_TOOLS_DIR ${MLIR_LLVM_BUILD_DIR}/bin)
set(MLIR_LLVM_LIBS_DIR ${MLIR_LLVM_BUILD_DIR}/lib)

set(LLVM_DIR
        "${MLIR_LLVM_BUILD_DIR}/lib/cmake/llvm")
message(STATUS "Using LLVM modules in: ${LLVM_DIR}")
find_package(LLVM REQUIRED CONFIG)
include(AddLLVM)
include(TableGen)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

set(MLIR_DIR
        "${MLIR_LLVM_BUILD_DIR}/lib/cmake/mlir")
message(STATUS "Using MLIR modules in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
find_package(MLIR REQUIRED CONFIG)
include(AddMLIR)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")

message(STATUS "LLVM ASSERTIONS is ${LLVM_ENABLE_ASSERTIONS}")
message(STATUS "LLVM EH is ${LLVM_ENABLE_EH}")
message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")

set(MLIR_SRC_INCLUDE_PATH ${MLIR_SOURCE_DIR}/include)
set(MLIR_BIN_INCLUDE_PATH ${MLIR_LLVM_BUILD_DIR}/tools/mlir/include)
message(STATUS "LLVM headers at: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR headers at: ${MLIR_INCLUDE_DIRS}")

#message(STATUS "Directory to search for LLVM libraries: ${LLVM_BUILD_LIBRARY_DIR}")
#link_directories(${LLVM_BUILD_LIBRARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

#message(STATUS "Project Source Dirs: ${PROJECT_SOURCE_DIR}")
#message(STATUS "Project Binary Dirs: ${PROJECT_BINARY_DIR}")
#include_directories(${PROJECT_SOURCE_DIR}/include)
#include_directories(${PROJECT_BINARY_DIR}/include)

function(scheduler_tablegen ofn)
    message(STATUS "Invoking tablegen " ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
    tablegen(MLIR
            ${ARGV}
            "-I${MLIR_SRC_INCLUDE_PATH}"
            "-I${MLIR_BIN_INCLUDE_PATH}"
            "-I${SCHEDULER_SRC_ROOT}")
    set(TABLEGEN_OUTPUT
            ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
            PARENT_SCOPE)
endfunction()

if (UNIX AND NOT APPLE)
install(FILES ${LLVM_LIBRARY_DIRS}/libMLIR.so.${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}
        ${LLVM_LIBRARY_DIRS}/libLLVM.so.${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib COMPONENT "${DEEPTOOLS_RUNTIME_COMPONENT}")
else()
install(FILES ${LLVM_LIBRARY_DIRS}/libMLIR.dylib
       ${LLVM_LIBRARY_DIRS}/libLLVM.dylib
       DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
message(STATUS "MLIR.cmake loading is finished")
