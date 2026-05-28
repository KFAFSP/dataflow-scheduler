#include "src/Bindings/Python/DialectKTDF.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
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
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAKTDFTokenType;
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
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAKTDFFifoSlotType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDFFifoSlotTypeGetTypeID;
  static constexpr const char* pyClassName = "FifoSlotType";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](ktdf::FifoType fifoType, int64_t numElements, MlirType elementType) {
          return PyFifoSlotType(
              PyMlirContext::forContext(mlirTypeGetContext(elementType)),
              mlirKTDFFifoSlotTypeGet(static_cast<uint32_t>(fifoType),
                                      numElements, elementType));
        },
        nb::arg("fifoType"), nb::arg("numElements"), nb::arg("elementType"));

    c.def_prop_ro("fifoType", [](PyFifoSlotType self) {
      return static_cast<ktdf::FifoType>(mlirKTDFFifoSlotTypeGetFifoType(self));
    });

    c.def_prop_ro("numElements", [](PyFifoSlotType self) {
      return mlirKTDFFifoSlotTypeGetNumElements(self);
    });

    c.def_prop_ro("elementType", [](PyFifoSlotType self) {
      return mlirKTDFFifoSlotTypeGetElementType(self);
    });
  }
};

struct PyMemorySpaceAttr : PyConcreteAttribute<PyMemorySpaceAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAKTDFMemorySpaceAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDFMemorySpaceAttrGetTypeID;
  static constexpr const char* pyClassName = "MemorySpaceAttr";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](ktdf::MemorySpace value, DefaultingPyMlirContext context) {
          return PyMemorySpaceAttr(
              context->getRef(),
              mlirKTDFMemorySpaceAttrGet(context->get(),
                                         static_cast<uint32_t>(value)));
        },
        nb::arg("value"), nb::arg("context").none() = nb::none());

    c.def_prop_ro("value", [](PyMemorySpaceAttr self) {
      return static_cast<ktdf::MemorySpace>(
          mlirKTDFMemorySpaceAttrGetValue(self));
    });
  }
};

struct PyLoopTypeAttr : PyConcreteAttribute<PyLoopTypeAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAKTDFLoopTypeAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDFLoopTypeAttrGetTypeID;
  static constexpr const char* pyClassName = "LoopTypeAttr";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](ktdf::LoopType value, DefaultingPyMlirContext context) {
          return PyLoopTypeAttr(
              context->getRef(),
              mlirKTDFLoopTypeAttrGet(context->get(),
                                      static_cast<uint32_t>(value)));
        },
        nb::arg("value"), nb::arg("context").none() = nb::none());

    c.def_prop_ro("value", [](PyLoopTypeAttr self) {
      return static_cast<ktdf::LoopType>(mlirKTDFLoopTypeAttrGetValue(self));
    });
  }
};

struct PyApplicableUnitsAttr : PyConcreteAttribute<PyApplicableUnitsAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAKTDFApplicableUnitsAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDFApplicableUnitsAttrGetTypeID;
  static constexpr const char* pyClassName = "ApplicableUnitsAttr";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](std::optional<std::vector<ktdf::ComponentType>> maybeValue,
           DefaultingPyMlirContext context) {
          const auto value = llvm::map_to_vector(
              maybeValue.value_or(std::vector<ktdf::ComponentType>()),
              [](auto x) { return static_cast<uint32_t>(x); });

          return PyApplicableUnitsAttr(
              context->getRef(),
              mlirKTDFApplicableUnitsAttrGet(context->get(), value.size(),
                                             value.data()));
        },
        nb::arg("value").none() = nb::none(),
        nb::arg("context").none() = nb::none());

    c.def_prop_ro("value", [](PyApplicableUnitsAttr self) {
      std::vector<ktdf::ComponentType> value(
          mlirKTDFApplicableUnitsAttrGetNumValues(self), {});
      for (auto&& [i, slot] : llvm::enumerate(value))
        slot = static_cast<ktdf::ComponentType>(
            mlirKTDFApplicableUnitsAttrGetValue(self, i));
      return value;
    });
  }
};

NB_MODULE(_ktirDialectsKTDF, m) {
  PyTokenType::bind(m);
  PyFifoSlotType::bind(m);

  PyMemorySpaceAttr::bind(m);
  PyLoopTypeAttr::bind(m);
  PyApplicableUnitsAttr::bind(m);
}