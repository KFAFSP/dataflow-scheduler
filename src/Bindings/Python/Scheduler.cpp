#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir-c/Bindings/Python/Interop.h>
#include <mlir-c/IR.h>
#include <nanobind/nanobind.h>

#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
#include "src/Analysis/ArchitectureGraph.hpp"
#include "src/Bindings/Python/DialectKTDF.hpp"

using namespace mlir;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
using namespace scheduler;
namespace nb = nanobind;

static const char kDumpDocstring[] =
    "Dumps a debug representation of the object to stderr.";

NB_MODULE(_ktirScheduler, m) {
  nb::class_<ArchitectureGraph> architectureGraph(
      m, "ArchitectureGraph",
      "Defines the memory resource graph of a target platform.");
  architectureGraph
      .def_static("createBuiltInSpyre", &ArchitectureGraph::createBuiltInSpyre,
                  "Create the architecture graph for the Spyre platform.")
      .def(
          "dump",
          [](const ArchitectureGraph& self) { self.dump(llvm::errs()); },
          kDumpDocstring)
      .def("__repr__",
           [](const ArchitectureGraph& self) {
             std::string result;
             {
               llvm::raw_string_ostream buffer(result);
               self.dump(buffer);
             }
             if (!result.empty() && result.back() == '\n')
               result.resize(result.size() - 1);
             return result;
           })
      .def_static(
          "getResourceFromFifoType",
          [](ktdf::FifoType fifo_type, bool is_producer_side) {
            return ArchitectureGraph::getResourceFromFifoType(fifo_type,
                                                              is_producer_side);
          },
          "Gets the resource associated with the specified end of a FIFO.",
          nb::arg("fifo_type"), nb::arg("is_producer_side"))
      .def(
          "getEdgeInfoForResources",
          [](const ArchitectureGraph& self, ArchitectureGraph::Resource source,
             ArchitectureGraph::Resource target) {
            return self.getEdgeInfoForResources(source, target);
          },
          "Obtains the edge between the two resources, if any.",
          nb::arg("source"), nb::arg("target"));

  nb::enum_<ArchitectureGraph::ResourceKind>(architectureGraph, "ResourceKind")
      .value("Memory", ArchitectureGraph::ResourceKind::Memory)
      .value("Compute", ArchitectureGraph::ResourceKind::Compute);

  nb::enum_<ArchitectureGraph::Resource>(architectureGraph, "Resource")
      .value("HBM", ArchitectureGraph::Resource::HBM)
      .value("LX", ArchitectureGraph::Resource::LX)
      .value("PE", ArchitectureGraph::Resource::PE)
      .value("SFP", ArchitectureGraph::Resource::SFP);

  nb::enum_<ArchitectureGraph::EdgeKind>(architectureGraph, "EdgeKind")
      .value("StorageTransfer", ArchitectureGraph::EdgeKind::StorageTransfer)
      .value("StreamingTransfer",
             ArchitectureGraph::EdgeKind::StreamingTransfer);

  nb::class_<ArchitectureGraph::ResourceNode>(architectureGraph, "ResourceNode")
      .def_ro("id", &ArchitectureGraph::ResourceNode::id)
      .def_ro("resource", &ArchitectureGraph::ResourceNode::resource);

  nb::class_<ArchitectureGraph::EdgeInfo>(architectureGraph, "EdgeInfo")
      .def_ro("source", &ArchitectureGraph::EdgeInfo::source)
      .def_ro("target", &ArchitectureGraph::EdgeInfo::target)
      .def_ro("edge_kind", &ArchitectureGraph::EdgeInfo::edge_kind)
      .def_ro("transfer_unit", &ArchitectureGraph::EdgeInfo::transfer_unit)
      .def_ro("fifo_type", &ArchitectureGraph::EdgeInfo::fifo_type)
      .def_ro("cost", &ArchitectureGraph::EdgeInfo::cost);
}
