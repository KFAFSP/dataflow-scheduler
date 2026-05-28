# RUN: python %s

from mlir_ktir.scheduler import ArchitectureGraph
import mlir_ktir.dialects.ktdf as ktdf

# ArchitectureGraph.getResourceFromFifoType

assert (
    ArchitectureGraph.getResourceFromFifoType(ktdf.FifoType.HbmToPe, True)
    == ArchitectureGraph.Resource.HBM
)
assert (
    ArchitectureGraph.getResourceFromFifoType(ktdf.FifoType.HbmToPe, False)
    == ArchitectureGraph.Resource.PE
)

# ArchitectureGraph.createBuiltInSpyre

spyre_graph = ArchitectureGraph.createBuiltInSpyre()

# ArchitectureGraph.getEdgeInfoForResources

sfp_to_lx = spyre_graph.getEdgeInfoForResources(
    ArchitectureGraph.Resource.SFP, ArchitectureGraph.Resource.LX
)

assert sfp_to_lx is not None
assert sfp_to_lx.edge_kind == ArchitectureGraph.EdgeKind.StreamingTransfer
assert sfp_to_lx.fifo_type == ktdf.FifoType.SfpToLxsu
assert sfp_to_lx.transfer_unit == ktdf.ComponentType.LXSU
