"""Simple MLIR-style diagnostic engine for rich Python diagnostics.

TODO: I would much rather have MLIR emit the diagnostics, since we are using
      the same mechanism. However, the MLIR Python bindings do not support
      generating anything but very simplistic error diagnostics.
"""

from collections import defaultdict
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
import io
from pathlib import Path
import sys
from typing import Final, Literal, Self, TypeAlias

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.segment import Segment
from rich.text import Text
from rich.theme import Theme

import mlir_ktir._mlir_libs._mlir as _mlir

Severity: TypeAlias = _mlir.ir.DiagnosticSeverity
Location: TypeAlias = _mlir.ir.Location


THEME: Final = Theme(
    {
        # Diagnostic severity labels.
        "severity.ERROR": "red bold",
        "severity.WARNING": "magenta bold",
        "severity.REMARK": "blue bold",
        "severity.NOTE": "cyan bold",
        # Diagnostic Location components.
        "location": "bold green",
        "location.file": "bold green",
        "location.line": "bold green",
        "location.column": "bold green",
        # Diagnostic messages.
        "diagnostic.ERROR": "bold",
        "diagnostic.WARNING": "bold",
        "diagnostic.REMARK": "bold",
        "diagnostic.NOTE": "bold",
    }
)
"""Default rich theme for the application."""


STDERR: Final = Console(stderr=True, theme=THEME)
"""Rich console that writes to `sys.stderr`."""


SOURCE_BUFFER: Final[dict[Path, tuple[str, ...]]] = dict()
"""Buffer for source files."""


@dataclass
class SourceRange:
    """Reference to a printable range within a source file."""

    filename: Path
    """Path to the file."""
    start_line: int = 1
    """Starting line index (1-based)."""
    start_col: int = 1
    """Starting column offset (1-based)."""
    end_line: int | None = None
    """End line index, or `None` for the same as the start."""
    end_col: int | None = None
    """End column index, or `None` for the same as the start."""

    def __rich_console__(
        self: Self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if (file := self.filename) != Path("-"):
            relative = file.absolute().relative_to(Path.cwd(), walk_up=True).__str__()
        else:
            relative = "<stdin>"

        yield Segment(relative.__str__(), console.get_style("location.file"))
        yield Segment(f":{self.start_line}", console.get_style("location.line"))
        yield Segment(f":{self.start_col}", console.get_style("location.column"))

    def _render_context(self: Self, console: Console) -> RenderResult:
        def _try_get_context() -> tuple[str, ...] | None:
            filename = self.filename.resolve()
            if (buffered := SOURCE_BUFFER.get(filename)) is None:
                try:
                    with io.open(self.filename, "r") as file:
                        buffered = SOURCE_BUFFER[filename] = tuple(file)
                except:
                    return None

            end_line = self.end_line or self.start_line + 1
            return buffered[self.start_line - 1 : end_line]

        if (context := _try_get_context()) is None:
            return

        for line_no, line in enumerate(context):
            prefix = f"{self.start_line + line_no:>5}"
            yield Text(prefix, style="location.line", end="")
            yield Segment(f"| {line}")

            col_span: int
            if line_no == 0:
                col_span = (self.end_col or self.start_col) - self.start_col - 1
                yield Segment(f"{' ' * len(prefix)}|{' ' * self.start_col}")
                yield Text("^", style="location.column")
            elif line_no == len(context):
                col_span = self.end_col or self.start_col
            else:
                col_span = len(line)

            if col_span > 1:
                yield Text(f"{'~' * (col_span)}", style="location.column")


def try_get_source_range(location: Location | None) -> SourceRange | None:
    """Tries to obtain a SourceRange from a Location.

    Returns:
        SourceRange | None: The first found SourceRange, or `None` if none is
            available.
    """

    if location is None:
        return None

    if location.is_a_callsite():
        return try_get_source_range(location.caller)

    if location.is_a_file():
        return SourceRange(
            Path(location.filename),
            location.start_line,
            location.start_col,
            location.end_line,
            location.end_col,
        )

    if location.is_a_fused():
        return next(
            filter(
                lambda x: x is not None, map(try_get_source_range, location.locations)
            ),
            None,
        )

    if location.is_a_name():
        return try_get_source_range(location.child_loc)

    return None


@dataclass
class Note:
    """Represents a note attached to a Diagnostic."""

    location: Location | None
    """Location at which this note applies."""

    message: str
    """Message string."""

    def __rich_console__(
        self: Self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if (source_range := try_get_source_range(self.location)) is not None:
            yield source_range
            yield Segment(": ", console.get_style("location"))

        yield Segment(f"note: ", console.get_style(f"severity.NOTE"))
        yield Segment(self.message, console.get_style(f"diagnostic.NOTE"))
        yield Segment("\n")

        if source_range is not None:
            yield from source_range._render_context(console)


@dataclass
class Diagnostic:
    """Represents a diagnostics message generated by the program."""

    location: Location | None
    """Location at which this diagnostic was emitted."""
    severity: Severity
    """Severity of this diagnostic."""
    message: str
    """Message string."""
    notes: tuple[Note, ...] = tuple()
    """Notes attached to this diagnostic (optional)."""

    def __rich_console__(
        self: Self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if (source_range := try_get_source_range(self.location)) is not None:
            yield source_range
            yield Segment(": ", console.get_style("location"))

        severity = self.severity.name
        yield Segment(
            f"{severity.lower()}: ", console.get_style(f"severity.{severity}")
        )
        yield Segment(self.message, console.get_style(f"diagnostic.{severity}"))
        yield Segment("\n")

        if source_range is not None:
            yield from source_range._render_context(console)

        for note in self.notes:
            yield note


DIAGNOSTIC_LEVEL_QUIET: Final = -1
"""Value for DIAGNOSTIC_LEVEL that suppresses all diagnostics."""

DIAGNOSTIC_LEVEL: Final = ContextVar[Severity | Literal[-1]](
    "diagnostic_level", default=Severity.WARNING
)
"""Configures a severity filter for the diagnostic emitter."""


def emit(diag: Diagnostic, /):
    """Emits a Diagnostic.

    Depending on the configured `DIAGNOSTIC_LEVEL`, the diagnostic may be
    silently swallowed.
    """

    if (level := DIAGNOSTIC_LEVEL.get()) == DIAGNOSTIC_LEVEL_QUIET:
        return

    if diag.severity.value <= level.value:
        STDERR.print(diag, crop=False)
