"""Implements a generic MLIR operation visitor."""

from collections.abc import Iterable
import functools

from frozendict import frozendict  # This will be a built-in in 3.15 anyways.
import inspect
import typing
from typing import Annotated, Any, Callable, Final, Literal, Self, Union

import mlir_ktir._mlir_libs._mlir as _mlir

GENERIC_OP_NAME: Final = "*"
"""Type that represents any MLIR operation."""

type GenericOpName = Literal["*"]
"""Type of the `GENERIC_OP_NAME` literal."""

type OpRef = _mlir.ir.Operation | _mlir.ir.OpView
"""Type of a reference to an MLIR operation."""

type OpType = type[_mlir.ir.Operation] | type[_mlir.ir.OpView]
"""Type of an MLIR operation adaptor class."""

type OpFilter = str | type[_mlir.ir.OpView] | type[_mlir.ir.Operation]
"""Type that constrains the type of an MLIR operation."""


def try_get_op_name(op_or_class: OpRef | OpType, /) -> str | GenericOpName:
    """Tries to get the fully qualified name of an operation or operation type.

    Args:
        op_or_class (OpRef | OpType): Operation instance, or class of the
            operation.

    Returns:
        str | None: Fully qualified name of the operation, `GENERIC_OP_NAME` if
            `op_or_class` is a generic adaptor.

    Raises:
        ValueError: `op_or_class` is an invalid `ir.OpView` subclass.
    """

    if isinstance(op_or_class, _mlir.ir.Operation):
        return op_or_class.name
    if isinstance(op_or_class, _mlir.ir.OpView):
        return op_or_class.operation.name
    if op_or_class in (_mlir.ir.Operation, _mlir.ir.OpView):
        return GENERIC_OP_NAME
    if isinstance(name := getattr(op_or_class, "OPERATION_NAME", False), str):
        return name

    raise ValueError(f"invalid 'ir.OpView' adaptor '{op_or_class}'")


def op_handler[T: Callable](handler: T, /) -> T:
    """Decorates a callable with a leading positional MLIR operation parameter.

    A callable with this decoration will be considered for operation-name based
    dispatch. The accepted operation names are determined by the type annotation
    on the leading non-self positional parameter:

    - A `ir.Operation` to handle any operation.
    - A subclass of `ir.OpView` to handle a particular operation.
    - A `typing.Annotated[ir.Operation, "op.name"]` to handle an operation
      with a specific name using the generic operation adaptor.
    - A `typing.Union` of accepted types.

    Raises:
        ValueError: At least one positional non-self parameter is required.
        ValueError: Op parameter must have a valid type annotation.
    """

    # Get the leading non-self positional parameter.
    parameters = list(inspect.signature(handler).parameters.values())
    if len(parameters) < 2 or parameters[1].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError("expected a positional op parameter")
    parameter = parameters[1]

    op_filters: list[OpFilter] = []

    def _collect_op_filters(annotation: Any):
        if typing.get_origin(annotation) is Union:
            # Collect all names in the union.
            for alternative in typing.get_args(annotation):
                _collect_op_filters(alternative)
            return

        if typing.get_origin(annotation) is Annotated:
            # Expect a generic operation adaptor with an op name string.
            param_type, op_name = typing.get_args(annotation)
            if param_type is not _mlir.ir.Operation:
                raise ValueError(
                    f"parameter '{parameter.name}' must accept 'ir.Operation'"
                )
            if not isinstance(op_name, str):
                raise ValueError(
                    f"parameter '{parameter.name}' must be annotated with operation name 'str'"
                )

            op_filters.append(op_name)
            return

        if annotation is _mlir.ir.Operation:
            # Expect the generic operation adaptor.
            op_filters.append(_mlir.ir.Operation)
            return

        if annotation is _mlir.ir.OpView:
            raise ValueError("use 'ir.Operation' instead of 'ir.OpView'")

        if issubclass(annotation, _mlir.ir.OpView):
            # Expect a typed operation adaptor, or the generic one.
            op_filters.append(annotation)
            return

        raise ValueError(f"unspported op type annotation '{annotation}'")

    _collect_op_filters(parameter.annotation)

    # Make a wrapper that ensures the proper downcasting is applied.
    @functools.wraps(handler)
    def _handler(self, op: OpRef, *args, **kwargs):
        # Invoke with specialized _mlir.ir.OpView adaptor.
        op = op.opview
        if op.__class__ in op_filters:
            return handler(self, op, *args, **kwargs)

        # Invoke with generic _mlir.ir.Operation adaptor.
        op = op.operation
        if op.name in op_filters:
            return handler(self, op, *args, **kwargs)

        # Handler was invoked with the wrong operation type.
        raise ValueError(
            f"invalid op '{op}', expected any of '{", ".join(str(op_filter) for op_filter in op_filters)}'"
        )

    setattr(_handler, "_op_handler", frozenset(op_filters))
    return _handler  # type: ignore


class OpDispatch[TResult]:
    """Base class for implementing MLIR operation name based dispatch.

    By declaring methods in this class with the `op_handler` decorator, they
    become registered with the dispatcher at class creation time. The base class
    offers a `_dispatch` method which can be used to invoke these handlers. If
    a trivial catch-all handler is registered, it will be used as the fallback.
    A custom fallback can also be provided by overriding the `_fallback` method.

    Dispatch is unique, i.e., there must be at most one handler for every
    operation name. If multiple handlers shall be invoked, use an `OpVisitor`.

    Args:
        handler_prefix (str | None): Name prefix for subclass members that
            should automatically be registered as handlers. Defaults to `None`.

    Raises:
        ValueError: More than one fallback handler was registered.
        ValueError: More than one handler registered for a particular op.
    """

    __op_handlers: frozendict[str, Callable[..., TResult]] = frozendict()

    def __init_subclass__(cls, /, handler_prefix: str | None = None) -> None:
        if handler_prefix is not None:
            # Automatically register all members with the handler_prefix.
            for name in [
                name for name in cls.__dict__ if name.startswith(handler_prefix)
            ]:
                setattr(cls, name, op_handler(cls.__dict__[name]))

        op_handlers = dict()

        # If the subclass declares a fallback handler, register that. Type
        # correctness should be given because the subclass must override our
        # own fallback handler, which is _not_ in the dict yet.
        if (fallback := cls.__dict__.get("_fallback")) is not None:
            op_handlers[GENERIC_OP_NAME] = fallback

        def _add_handler(op_filter: OpFilter, handler: Callable[..., TResult]):
            op_name = (
                op_filter if isinstance(op_filter, str) else try_get_op_name(op_filter)
            )
            if op_name in op_handlers:
                if op_name == GENERIC_OP_NAME:
                    raise ValueError("more than one fallback handler registered")
                else:
                    raise ValueError(
                        f"more than one handler registered for '{op_name}'"
                    )

            op_handlers[op_name] = handler

        # Register all handler methods.
        for member in cls.__dict__.values():
            op_filters = getattr(member, "_op_handler", False)
            if not isinstance(op_filters, frozenset):
                continue

            # Register this handler for all its supported op types.
            for op_filter in op_filters:
                _add_handler(op_filter, member)

        if (fallback := cls.__op_handlers.get(GENERIC_OP_NAME)) is not None:
            # Install the fallback handler to the class body.
            cls._fallback = fallback
        else:
            # Install our fallback handler to the lookup.
            op_handlers[GENERIC_OP_NAME] = OpDispatch._fallback

        cls.__op_handlers = frozendict(op_handlers)

    def _fallback(self: Self, op: _mlir.ir.Operation, /, *args, **kwargs) -> TResult:
        """Throws a ValueError that indicates the operation is unsupported."""

        raise ValueError("unsupported operation '{op_name}'")

    def _dispatch(self: Self, op: OpRef, /, *args, **kwargs) -> TResult:
        """Dispatches the operation to its unique handler."""

        # Dispatch by name.
        if (handler := self.__op_handlers.get(op.operation.name)) is not None:
            # NOTE: If the wrapper throws a ValueError, this likely means that
            #       the desired ir.OpView specialization was not registered or
            #       replaced for the associated operation name!
            return handler(self, op, *args, **kwargs)

        return self._fallback(op.operation, *args, **kwargs)


class OpVisitor[THandler, TResult = None]:
    """Base class for implementing MLIR operation visitors.

    By declaring methods in this class with the `op_handler` decorator, they
    become registered with the visitor at class creation time. The base class
    offers a `visit` method which can be used to invoke these handlers.

    The visitor preserves the order in which the handlers are registered in,
    allowing multiple handlers to be registered for every op. Trivial catch-all
    handlers become registered for all ops.

    Args:
        handler_prefix (str | None, optional): Name prefix for subclass members
            that should automatically be registered as handlers.
            Defaults to `visit_`.
    """

    __op_handlers: frozendict[str, tuple[Callable[..., THandler]]] = frozendict()

    def __init_subclass__(cls, /, handler_prefix: str | None = "visit_") -> None:
        if handler_prefix is not None:
            # Automatically register all members with the handler_prefix.
            for name in [
                name for name in cls.__dict__ if name.startswith(handler_prefix)
            ]:
                setattr(cls, name, op_handler(cls.__dict__[name]))

        op_handlers = dict()

        def _add_handler(op_filter: OpFilter, handler: Callable[..., THandler]):
            op_name = (
                op_filter if isinstance(op_filter, str) else try_get_op_name(op_filter)
            )

            # Add this handler to the tuple, initializing it with all the
            # previously registered fallback handlers.
            fallbacks = op_handlers.get(GENERIC_OP_NAME, ())
            op_handlers[op_name] = op_handlers.get(op_name, fallbacks) + (handler,)

            if op_name == GENERIC_OP_NAME:
                # Add this handler to all already known filters.
                known_names = set(op_handlers)
                known_names.remove(op_name)

                def _handler(self, op: OpRef, *args, **kwargs):
                    handler(self, op.operation, *args, **kwargs)

                for name in known_names:
                    op_handlers[name] = op_handlers[name] + (_handler,)

        # Register all handler methods.
        for member in cls.__dict__.values():
            op_filters = getattr(member, "_op_handler", False)
            if not isinstance(op_filters, frozenset):
                # This is not a handler.
                continue

            # Register this handler for all its supported op types.
            for op_filter in op_filters:
                _add_handler(op_filter, member)

        cls.__op_handlers = frozendict(op_handlers)

    def _reduce(self: Self, results: Iterable[THandler], /) -> TResult:
        """Reduces the results of the handlers to a visit result.

        Args:
            results (Iterable[THandler]): Handler results, which may be lazily
                evaluated.
        """

        # Force eager evaluation of all handlers.
        for result in results:
            pass

        # NOTE: You have to override this method if you want TResult != None.
        return None  # type: ignore

    def visit(self: Self, op: OpRef, /, *args, **kwargs) -> TResult:
        """Visits the operation."""

        fallbacks = self.__op_handlers.get(GENERIC_OP_NAME, ())
        handlers = self.__op_handlers.get(op.operation.name, fallbacks)

        # Dispatch by name and reduce the results.
        return self._reduce(
            map(lambda handler: handler(self, op, *args, **kwargs), handlers)
        )


class IRVisitor(OpVisitor[_mlir.ir.WalkResult, _mlir.ir.WalkResult]):
    """Base class for implementing MLIR IR visitors.

    Specializes the `OpVisitor` by replacing the `visit` method with a call to
    MLIR's `ir.Operation.walk` method, which recursively visits all operations
    nested beneath the root. To control walking, handlers must return an
    `ir.WalkResult`. If multiple handlers are invoked, the result is interpreted
    as follows:

    - If at least one `ir.WalkResult.INTERRUPT` is received, the current op's
      handlers are completed, and the visitation is interrupted.
    - If at least one `ir.WalkResult.ADVANCE` is received, the walk continues
      with the next operation in the next nesting level.
    - If all results are `ir.WalkResult.SKIP`, the next nesting level is
      skipped.

    Args:
        handler_prefix (str | None, optional): Name prefix for subclass members
            that should automatically be registered as handlers.
            Defaults to `visit_`.
    """

    # FIXME: This is a workaround for MLIR not returning the WalkResult.
    _interrupted: bool = False

    def _reduce(
        self: Self, results: Iterable[_mlir.ir.WalkResult | None], /
    ) -> _mlir.ir.WalkResult:
        combined = _mlir.ir.WalkResult.SKIP

        # Force eager evaluation of all handlers.
        for result in list(results):
            # For ease of use, take None to mean ADVANCE.
            result = result or _mlir.ir.WalkResult.ADVANCE

            if result == _mlir.ir.WalkResult.INTERRUPT:
                self._interrupted = True
                return result
            if result == _mlir.ir.WalkResult.ADVANCE:
                combined = result

        return combined

    def visit(
        self: Self,
        root: OpRef | _mlir.ir.Module,
        /,
        walk_order: _mlir.ir.WalkOrder = _mlir.ir.WalkOrder.POST_ORDER,
    ) -> _mlir.ir.WalkResult:
        """Recursively visits the root and its descendants."""

        if isinstance(root, _mlir.ir.Module):
            root = root.operation

        # FIXME: MLIR's bindings should return the WalkResult here!
        self._interrupted = False
        root.walk(super().visit, walk_order)
        return (
            _mlir.ir.WalkResult.ADVANCE
            if not self._interrupted
            else _mlir.ir.WalkResult.INTERRUPT
        )


# def walk(
#     root: OpRef,
#     *handlers: Callable[..., _mlir.ir.WalkResult],
#     walk_order: _mlir.ir.WalkOrder = _mlir.ir.WalkOrder.POST_ORDER,
# ) -> _mlir.ir.WalkResult:
#     visitor_class = type(
#         "WalkVisitor",
#         (IRVisitor,),
#         {
#             f"visit_{number}": op_handler(handler)
#             for number, handler in enumerate(handlers)
#         },
#     )

#     return visitor_class().visit(root, walk_order)
