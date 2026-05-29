include(AddKTIRPython)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(KTIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ktir")
set(KTIR_PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/Bindings/Python")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KTIRSchedulerPythonSources
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  SOURCES
    scheduler/__init__.py
    simulator/diagnostic.py
    simulator/scev.py
    simulator/validator.py
    simulator/visitor.py
    __init__.py
  ADD_TO_PARENT KTIRPythonSources
)
declare_mlir_python_sources(KTIRSchedulerPythonExtensions
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  SOURCES
    _mlir_libs/_ktir/py.typed
  ADD_TO_PARENT KTIRPythonExtensions
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AgenOps.td
  SOURCES 
    dialects/agen.py
  DIALECT_NAME agen
  GEN_ENUM_BINDINGS
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/DataflowOps.td
  SOURCES 
    dialects/dataflow.py
  DIALECT_NAME dataflow
  GEN_ENUM_BINDINGS
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/KTDFOps.td
  SOURCES 
    dialects/ktdf.py
  DIALECT_NAME ktdf
  GEN_ENUM_BINDINGS
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/UniformOps.td
  SOURCES 
    dialects/uniform.py
  DIALECT_NAME uniform
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/VectorChainOps.td
  SOURCES 
    dialects/vectorchain.py
  DIALECT_NAME vectorchain
  GEN_ENUM_BINDINGS
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(KTIRSchedulerPythonExtensions.Main
  MODULE_NAME _ktir
  ADD_TO_PARENT KTIRSchedulerPythonExtensions
  SOURCES
    KTIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPILinalg
    MLIRCAPIMath
    MLIRCAPISCF
    MLIRCAPITensor
    MLIRCAPITransforms
    KTIRKtdpCAPI
    KTIRAgenCAPI
    KTIRDataflowCAPI
    KTIRKTDFCAPI
    KTIRRegisterEverythingCAPI
    KTIRUniformCAPI
    KTIRVectorChainCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_extension(KTIRSchedulerPythonExtensions.Dialects.KTDF.Nanobind
  MODULE_NAME _ktirDialectsKTDF
  ADD_TO_PARENT KTIRPythonSources.Dialects.ktdf
  ROOT_DIR "${KTIR_PYTHON_SOURCE_DIR}"
  SOURCES
    DialectKTDF.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    KTIRKTDFCAPI
)

declare_mlir_python_extension(KTIRSchedulerPythonExtensions.Scheduler
  MODULE_NAME _ktirScheduler
  ADD_TO_PARENT KTIRSchedulerPythonExtensions
  ROOT_DIR "${KTIR_PYTHON_SOURCE_DIR}"
  SOURCES
    Scheduler.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    KTIRSchedulerAnalysis
  PRIVATE_LINK_LIBS
    LLVMSupport
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  add_custom_target(KTIRSchedulerPythonTypeStubs)

  function(add_ktir_python_type_stubs extension)
    cmake_parse_arguments(ARG "" "ADD_TO_PARENT" "DEPENDS_TARGETS;OUTPUTS;IMPORT_PATHS" ${ARGN})

    get_target_property(_extension_sources ${extension} INTERFACE_SOURCES)
    get_target_property(_extension_module_name ${extension} mlir_python_EXTENSION_MODULE_NAME)

    mlir_generate_type_stubs(
      MODULE_NAME "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs.${_extension_module_name}"
      DEPENDS_TARGETS "${ARG_DEPENDS_TARGETS}"
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
      OUTPUTS "${ARG_OUTPUTS}"
      DEPENDS_TARGET_SRC_DEPS "${_extension_sources}"
      IMPORT_PATHS "${KTIR_PYTHON_PACKAGE_DIR}/..;${ARG_IMPORT_PATHS}"
    )
    add_dependencies(KTIRSchedulerPythonTypeStubs ${NB_STUBGEN_CUSTOM_TARGET})

    set(_generated_outputs "${ARG_OUTPUTS}")
    list(TRANSFORM _generated_outputs PREPEND "_mlir_libs/")

    declare_mlir_python_sources(
      ${extension}.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT ${ARG_ADD_TO_PARENT}
      SOURCES "${_generated_outputs}"
    )
  endfunction()

  add_ktir_python_type_stubs(KTIRSchedulerPythonExtensions.Main
    ADD_TO_PARENT KTIRSchedulerPythonSources
    DEPENDS_TARGETS
      KTIRPythonModules.extension._mlir.dso
      KTIRPythonModules.extension._ktir.dso
    OUTPUTS
      _ktir.pyi
  )

  add_ktir_python_type_stubs(KTIRSchedulerPythonExtensions.Dialects.KTDF.Nanobind
    ADD_TO_PARENT KTIRSchedulerPythonSources
    DEPENDS_TARGETS
      KTIRPythonModules.extension._mlir.dso
      KTIRPythonModules.extension._ktirDialectsKTDF.dso
    OUTPUTS
      _ktirDialectsKTDF.pyi
  )

  add_ktir_python_type_stubs(KTIRSchedulerPythonExtensions.Scheduler
    ADD_TO_PARENT KTIRSchedulerPythonSources
    DEPENDS_TARGETS
      KTIRPythonModules.extension._mlir.dso
      KTIRPythonModules.extension._ktirScheduler.dso
    OUTPUTS
      _ktirScheduler.pyi
  )
endif()

################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.index
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.scf
  KTIRPythonSources
  KTIRPythonExtensions
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _source_components MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_common_capi_library(KTIRPythonCAPI
  INSTALL_COMPONENT KTIRPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${KTIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.." # NOTE: This is fragile.
  DECLARED_SOURCES ${_source_components}
)

ktir_python_fix_rpath(KTIRPythonCAPI 
  RELATIVE_INSTALL_ROOT "../../../.." # NOTE: This is fragile.
)

add_mlir_python_modules(KTIRPythonModules
  ROOT_PREFIX "${KTIR_PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    KTIRPythonCAPI
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  add_dependencies(KTIRPythonModules KTIRSchedulerPythonTypeStubs)
endif()
