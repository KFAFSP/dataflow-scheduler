"""Implements IR validation."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final, Self, cast

import mlir_ktir.dialects.ktdf as ktdf

from mlir_ktir.scheduler import ArchitectureGraph
from mlir_ktir.simulator import visitor
from mlir_ktir.simulator.diagnostic import Diagnostic, emit, Note, Severity
from mlir_ktir.simulator.visitor import IRVisitor, OpRef

import mlir_ktir._mlir_libs._mlir as _mlir


@dataclass
class ValidationResult:
    validator: type[IRValidator]
    """Generating validator class."""
    errors: tuple[Diagnostic, ...] = tuple()
    """Generated errors."""
    warnings: tuple[Diagnostic, ...] = tuple()
    """Generated warnings."""

    def __bool__(self: Self) -> bool:
        """Obtains a value indicating whether no errors were found."""

        return len(self.errors) == 0

    def __iter__(self: Self) -> Iterator[Diagnostic]:
        """Iterates over all generated diagnostics."""

        yield from self.errors
        yield from self.warnings

    def __len__(self: Self) -> int:
        """Gets the total number of generated diagnostics."""

        return len(self.errors) + len(self.warnings)


class IRValidator(IRVisitor):
    """Base class for an IR validator.

    An IR validator is an `IRVisitor` that accumulates errors and warnings into
    its internal state, reporting them as they are generated. Validators should
    be transient objects, directly invoking `run` on the constructed object.
    """

    _errors: list[Diagnostic] = list()
    """List of generated errors."""
    _warnings: list[Diagnostic] = list()
    """List of generated warnings."""

    def __init_subclass__(cls, /, handler_prefix: str | None = "check_") -> None:
        return super().__init_subclass__(handler_prefix)

    def remark(
        self: Self,
        loc_or_op: _mlir.ir.Location | OpRef,
        /,
        message: str,
        notes: Iterable[Note] = tuple(),
    ):
        if not isinstance(loc_or_op, _mlir.ir.Location):
            loc_or_op = loc_or_op.location

        emit(Diagnostic(loc_or_op, Severity.REMARK, message, tuple(notes)))

    def warn(
        self: Self,
        loc_or_op: _mlir.ir.Location | visitor.OpRef,
        /,
        message: str,
        notes: Iterable[Note] = tuple(),
    ):
        if not isinstance(loc_or_op, _mlir.ir.Location):
            loc_or_op = loc_or_op.location

        diag = Diagnostic(loc_or_op, Severity.WARNING, message, tuple(notes))
        self._warnings.append(diag)
        emit(diag)

    def fail(
        self: Self,
        loc_or_op: _mlir.ir.Location | visitor.OpRef,
        /,
        message: str,
        notes: Iterable[Note] = tuple(),
    ):
        if not isinstance(loc_or_op, _mlir.ir.Location):
            loc_or_op = loc_or_op.location

        diag = Diagnostic(loc_or_op, Severity.ERROR, message, tuple(notes))
        self._errors.append(diag)
        emit(diag)

    def run(self: Self, root: visitor.OpRef | _mlir.ir.Module, /) -> ValidationResult:
        """Runs the IR validator on the given root."""

        self._errors.clear()
        self._warnings.clear()
        self.visit(root)
        return ValidationResult(
            self.__class__, tuple(self._errors), tuple(self._warnings)
        )


class TransferValidator(IRValidator):
    """Validates that all memory transfers follow direct paths on the
    architecture graph.
    """

    architecture_graph: Final[ArchitectureGraph]
    """Architecture graph to validate against."""

    def __init__(self: Self, architecture_graph: ArchitectureGraph):
        self.architecture_graph = architecture_graph

    def _get_resource(
        self: Self,
        declaration: (
            ktdf.MemorySpace
            | ktdf.FifoType
            | _mlir.ir.Attribute
            | _mlir.ir.Value
            | _mlir.ir.Type
        ),
        is_producer_side: bool = False,
        /,
    ) -> ArchitectureGraph.Resource | None:
        SPACE_TO_RESOURCE: Final[dict] = {
            ktdf.MemorySpace.SFPRegisterFile: ArchitectureGraph.Resource.SFP,
            ktdf.MemorySpace.LX: ArchitectureGraph.Resource.LX,
            ktdf.MemorySpace.HBM: ArchitectureGraph.Resource.HBM,
        }

        if isinstance(declaration, _mlir.ir.Value):
            declaration = declaration.type

        if isinstance(declaration, _mlir.ir.MemRefType):
            if declaration.memory_space is None:
                return None

            declaration = declaration.memory_space

        if isinstance(declaration, ktdf.FifoSlotType):
            declaration = declaration.fifoType

        if isinstance(declaration, ktdf.FifoType):
            return self.architecture_graph.getResourceFromFifoType(
                declaration, is_producer_side
            )

        if isinstance(declaration, ktdf.MemorySpaceAttr):
            declaration = declaration.value

        return SPACE_TO_RESOURCE.get(declaration)

    def _check_transfer(
        self: Self,
        loc: _mlir.ir.Location,
        source: ArchitectureGraph.Resource,
        target: ArchitectureGraph.Resource,
    ):
        edge = self.architecture_graph.getEdgeInfoForResources(source, target)
        if edge is None:
            self.fail(loc, f"no transfer path from {source.name} to {target.name}")

    def check_ktdf_fifo_allocate(self: Self, op: ktdf.FifoAllocateOp, /):
        for type in cast(Iterable[ktdf.FifoSlotType], op.results.types):
            self._check_transfer(
                op.location,
                self.architecture_graph.getResourceFromFifoType(type.fifoType, True),
                self.architecture_graph.getResourceFromFifoType(type.fifoType, False),
            )

    def check_ktdf_data_transfer(self: Self, op: ktdf.DataTransferOp, /):
        if (source := self._get_resource(op.source, True)) is None:
            self.warn(op.source.location, "unknown memory resource")
        if (target := self._get_resource(op.destination)) is None:
            self.warn(op.destination.location, "uknown memory resource")
        if source and target:
            self._check_transfer(op.location, source, target)
