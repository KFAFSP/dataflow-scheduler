"""Implements a symbolic integer scalar evolution analysis.

TODO: Handle block argumsnts.
      This implementation does not handle block arguments at the moment, partly
      because MLIR is not giving us access to the `RegionBranchOpInterface` in
      its bindings.
      Additionally, this analysis does not implement add recurrences, as the
      structural control flow in MLIR is going to yield us the IVs as symbols
      anyways.

TODO: Add an assumptions system.
      When a symbol is created, it should be populated with assumptions. In
      particular, we would like to have more concrete (more than non-negative)
      bounds on loop IVs.
"""

from typing import Final, Iterable, Self, TypeAlias, cast, overload

import sympy
import sympy.core.expr

import mlir_ktir._mlir_libs._mlir as _mlir
import mlir_ktir.dialects.arith as arith
import mlir_ktir.dialects.index as index
import mlir_ktir.dialects.ktdp as ktdp
import mlir_ktir.dialects.scf as scf

import mlir_ktir.simulator.visitor as visitor

# NOTE: These can't be `type alias =` aliases because then they would not be
#       testable or invocable.
Symbol: TypeAlias = sympy.Symbol
Expression: TypeAlias = sympy.core.expr.Expr


class ScalarEvolution(visitor.OpDispatch[Expression], handler_prefix="visit_"):
    """Implements a (symbolic) static value analysis of integer variables."""

    KTDP_COMPUTE_TILE_ID: Final = sympy.Symbol(
        ktdp.GetComputeTileIdOp.OPERATION_NAME, integer=True, negative=False
    )
    """Symbol for the KTDP compute tile id."""

    INDEX_BITS: Final = sympy.Symbol(
        index.SizeOfOp.OPERATION_NAME, integer=True, positive=True
    )
    """Symbol for the index type size in bits."""

    SUPPORTED_TYPES: Final = (_mlir.ir.IndexType, _mlir.ir.IntegerType)
    """List of types supported by this analysis."""

    _symbols: dict[Symbol, _mlir.ir.Value] = {}
    """Maps symbols to their defining values."""

    _cache: dict[_mlir.ir.Value, Expression] = {}
    """Maps values to their symbolic expressions."""

    @overload
    def invalidate(self: Self, /) -> bool:
        """Invalidates all analysis results.

        Returns:
            bool: Always `True`.
        """
        ...

    @overload
    def invalidate(self: Self, value: _mlir.ir.Value, /) -> bool:
        """Invalidates the analysis result for a single value.

        Note that transitive dependencies (derived values) are not invalidated!

        Returns:
            bool: `True` if the value was unmapped, `False` if it is unknown.
        """
        ...

    def invalidate(self: Self, value_or_none: _mlir.ir.Value | None = None, /) -> bool:
        if value_or_none is None:
            # Remove all mappings.
            self._symbols.clear()
            self._cache.clear()
            return True

        if not (expr := self._cache.pop(value_or_none, False)):
            # Value was not mapped.
            return False

        # Also remove the symbol, if the value has one.
        if isinstance(expr, Symbol):
            self._symbols.pop(expr, False)
        return True

    def symbol(self: Self, value: _mlir.ir.Value, /, name: str | None = None) -> Symbol:
        """Declares value as a symbol.

        Args:
            value (Value): Value to declare as a symbol.
            name (str | None, optional): Unique name of the symbol, or `None` to
                make a unique symbol with a description. Defaults to `None`.

        Raises:
            ValueError: Attempted to redeclare the value.
            ValueError: Attempted to redeclare the symbol name.
        """

        RESERVED_SYMBOLS = (self.KTDP_COMPUTE_TILE_ID, self.INDEX_BITS)

        if value in self._cache:
            raise ValueError("value '{}' cannot be redeclared".format(value.__str__()))

        if name is None:
            # Derive a unique symbol for this value, with an MLIR description.
            symbol = sympy.Dummy(value.__str__())
        else:
            # Ensure that the symbol name is unique.
            symbol = Symbol(name)
            if symbol in self._symbols or symbol in RESERVED_SYMBOLS:
                raise ValueError("symbol '{}' cannot be redeclared".format(name))

        self._symbols[symbol] = value
        self._cache[value] = symbol
        return symbol

    @overload
    def visit(self: Self, value: _mlir.ir.Value, /) -> Expression:  # type: ignore
        """Obtains a symbolic expression for the input value.

        Raises:
            ValueError: Value has an unsupported type.
        """
        ...

    @overload
    def visit(self: Self, op: visitor.OpRef, /) -> tuple[Expression | None, ...]:
        """Obtains a tuple of symbolic expressions for the op result values.

        If a result value can't be analyzed, `None` is returned instead.
        """
        ...

    def visit(
        self: Self, value_or_op: _mlir.ir.Value | visitor.OpRef, /
    ) -> Expression | tuple[Expression | None, ...]:
        if not isinstance(value_or_op, _mlir.ir.Value):

            def _map(result: _mlir.ir.OpResult) -> Expression | None:
                return (
                    self.visit(result) if result.type in self.SUPPORTED_TYPES else None
                )

            return tuple(map(_map, value_or_op.results))

        if (expr := self._cache.get(value_or_op)) is not None:
            return expr

        # Check that the variable type is SCEV-able.
        if not isinstance(value_or_op.type, self.SUPPORTED_TYPES):
            raise ValueError("unsupported type '{}'".format(str(value_or_op.type)))

        def _visit(value: _mlir.ir.Value) -> Expression:
            if isinstance(value, _mlir.ir.OpResult):
                return self._dispatch(value_or_op.owner, value_or_op)

            # TODO: Handle block arguments.
            return self._fallback(None, value)

        # Visit the value and cache the result.
        result = self._cache[value_or_op] = _visit(value_or_op)
        return result

    def get_width(
        self: Self, value_or_type: _mlir.ir.Value | _mlir.ir.Type, /
    ) -> Expression | None:
        """Gets the bit width of the specified value or type.

        Returns:
            Expression | None: Bit width expression, or `None` if not available.
        """

        if isinstance(value_or_type, _mlir.ir.Value):
            value_or_type = value_or_type.type
        if isinstance(value_or_type, _mlir.ir.IndexType):
            return self.INDEX_BITS
        if isinstance(value_or_type, _mlir.ir.IntegerType):
            return sympy.S(value_or_type.width)

        return None

    def __iter__(self: Self) -> Iterable[tuple[_mlir.ir.Value, Expression]]:
        """Iterate over the cached value-to-expression mappings."""

        return self._cache.items().__iter__()

    def __len__(self: Self) -> int:
        """Gets the number of cached value-to-expression mappings."""

        return len(self._cache)

    @overload
    def __getitem__(self: Self, value: _mlir.ir.Value, /) -> Expression:  # type: ignore
        """Obtains a symbolic expression for the input value.

        Raises:
            ValueError: Value has an unsupported type.
        """
        ...

    @overload
    def __getitem__(self: Self, op: visitor.OpRef, /) -> tuple[Expression | None, ...]:
        """Obtains a tuple of symbolic expressions for the op result values.

        If a result value can't be analyzed, `None` is returned instead.
        """
        ...

    def __getitem__(
        self: Self, value_or_op: _mlir.ir.Value | visitor.OpRef, /
    ) -> Expression | tuple[Expression | None, ...]:
        return self.visit(value_or_op)

    def __contains__(self, value: _mlir.ir.Value, /) -> bool:
        """Determines whether the value has a cached expression."""
        return value in self._cache

    # --------------------------------------------------------------------------
    # OpDispatch handlers
    # --------------------------------------------------------------------------

    def _fallback(
        self: Self, op: visitor.OpRef | None, value: _mlir.ir.Value, /
    ) -> Expression:
        # There was no handler, or the value is a block argument. Either way, we
        # cannot analyze this value, because we don't have access to the branch
        # interfaces in Python...

        # NOTE: sympy.Dummy symbols are always unique, which means we don't need
        #       to have a unique name for them.
        return self.symbol(value)

    def visit_ktdp_compute_tile_id(
        self: Self, op: ktdp.GetComputeTileIdOp, result: _mlir.ir.OpResult, /
    ) -> Symbol:
        return self.KTDP_COMPUTE_TILE_ID

    def visit_index_size_of(
        self: Self, op: index.SizeOfOp, result: _mlir.ir.OpResult, /
    ) -> Symbol:
        return self.INDEX_BITS

    def visit_const(
        self: Self,
        op: arith.ConstantOp | index.ConstantOp,
        result: _mlir.ir.OpResult,
        /,
    ) -> Expression:
        return sympy.S(cast(_mlir.ir.IntegerAttr, op.value).value)

    def visit_arith_bitcast(
        self: Self, op: arith.BitcastOp, result: _mlir.ir.OpResult, /
    ) -> Expression:
        # arith only supports signless integer types, so there's nothing we can
        # infer at this point.
        return self.visit(op.in_)

    def visit_signed_cast(
        self: Self, op: index.CastSOp | arith.IndexCastOp, result: _mlir.ir.OpResult, /
    ) -> Expression:
        in_width = self.get_width(op.operands[0])
        out_width = self.get_width(result)

        # TODO: Handle overflow.
        # Sign-extension is assumed to be value-preserving.
        if in_width is not None and out_width is not None and in_width <= out_width:  # type: ignore
            return self.visit(op.operands[0])

        return self.symbol(result)

    def visit_addi(
        self: Self, op: index.AddOp | arith.AddIOp, result: _mlir.ir.OpResult, /
    ) -> Expression:
        # TODO: Handle overflow.
        return self.visit(op.lhs) + self.visit(op.rhs)  # type: ignore

    def visit_subi(
        self: Self, op: index.SubOp | arith.SubIOp, result: _mlir.ir.OpResult, /
    ) -> Expression:
        # TODO: Handle overflow.
        return self.visit(op.lhs) - self.visit(op.rhs)  # type: ignore

    def visit_muli(
        self: Self, op: index.MulOp | arith.MulIOp, result: _mlir.ir.OpResult, /
    ) -> Expression:
        # TODO: Handle overflow.
        return self.visit(op.lhs) * self.visit(op.rhs)  # type: ignore
