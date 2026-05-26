#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir-c/IR.h>
#include <nanobind/nanobind.h>

#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
#include "src/CAPI/KTDF.h"

using namespace mlir;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
namespace nb = nanobind;

struct PyTokenType : PyConcreteType<PyTokenType> {
  static constexpr IsAFunctionTy isaFunction = mlirKTDFTypeIsATokenType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDFTokenTypeGetTypeID;
  static constexpr const char* pyClassName = "TokenType";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyTokenType(context->getRef(),
                             mlirKTDFTokenTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct PyFifoSlotType : PyConcreteType<PyFifoSlotType> {
  static constexpr IsAFunctionTy isaFunction = mlirKTDFTypeIsAFifoSlotType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDFFifoSlotTypeGetTypeID;
  static constexpr const char* pyClassName = "FifoSlotType";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](uint32_t fifoType, int64_t numElements, MlirType elementType) {
          return PyFifoSlotType(
              PyMlirContext::forContext(mlirTypeGetContext(elementType)),
              mlirKTDFFifoSlotTypeGet(fifoType, numElements, elementType));
        },
        nb::arg("fifoType"), nb::arg("numElements"), nb::arg("elementType"));

    c.def_prop_ro("fifoType", [](PyFifoSlotType self) {
      return mlirKTDFFifoSlotTypeGetFifoType(self);
    });

    c.def_prop_ro("numElements", [](PyFifoSlotType self) {
      return mlirKTDFFifoSlotTypeGetNumElements(self);
    });

    c.def_prop_ro("elementType", [](PyFifoSlotType self) {
      return mlirKTDFFifoSlotTypeGetElementType(self);
    });
  }
};

struct PyApplicableUnitsAttr : PyConcreteAttribute<PyApplicableUnitsAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirKTDFAttributeIsApplicableUnitsAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDFApplicableUnitsAttrGetTypeID;
  static constexpr const char* pyClassName = "ApplicableUnitsAttr";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](std::optional<std::vector<uint32_t>> maybeComponentTypes,
           DefaultingPyMlirContext context) {
          const ArrayRef<uint32_t> componentTypes =
              maybeComponentTypes.has_value()
                  ? ArrayRef<uint32_t>(*maybeComponentTypes)
                  : ArrayRef<uint32_t>{};
          return PyApplicableUnitsAttr(
              context->getRef(), mlirKTDFApplicableUnitsAttrGet(
                                     context->get(), componentTypes.size(),
                                     componentTypes.data()));
        },
        nb::arg("componentTypes").none() = nb::none(),
        nb::arg("context").none() = nb::none());

    c.def_prop_ro("componentTypes", [](PyApplicableUnitsAttr self) {
      std::vector<uint32_t> componentTypes(
          mlirKTDFApplicableUnitsAttrGetNumComponents(self), 0);
      for (auto&& [i, slot] : llvm::enumerate(componentTypes))
        slot = mlirKTDFApplicableUnitsAttrGetComponent(self, i);
      return componentTypes;
    });
  }
};

NB_MODULE(_ktirDialectsKTDF, m) {
  PyTokenType::bind(m);
  PyFifoSlotType::bind(m);

  PyApplicableUnitsAttr::bind(m);
}