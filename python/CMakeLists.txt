include(AddKTIRPython)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(KTIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ktir")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KTIRSchedulerPythonSources
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  SOURCES
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
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/DataflowOps.td
  SOURCES 
    dialects/dataflow.py
  DIALECT_NAME dataflow
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/KTDFOps.td
  SOURCES 
    dialects/ktdf.py
  DIALECT_NAME ktdf
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

if(MLIR_PYTHON_STUBGEN_ENABLED)
  get_target_property(_ktirextension_srcs KTIRSchedulerPythonExtensions.Main INTERFACE_SOURCES)
  mlir_generate_type_stubs(
    MODULE_NAME mlir_ktir._mlir_libs._ktir
    DEPENDS_TARGETS
      KTIRPythonModules.extension._mlir.dso
      KTIRPythonModules.extension._ktir.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
    OUTPUTS
      _ktir.pyi
    DEPENDS_TARGET_SRC_DEPS "${_ktirextension_srcs}"
    IMPORT_PATHS "${KTIR_PYTHON_PACKAGE_DIR}/.."
  )
  set(_ktir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  declare_mlir_python_sources(
    KTIRSchedulerPythonExtensions.Main.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    ADD_TO_PARENT KTIRSchedulerPythonSources
    SOURCES
      _mlir_libs/_ktir.pyi
  )
endif()

################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
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
  add_dependencies(KTIRPythonModules ${_ktir_typestub_gen_target})
endif()
