from contextlib import contextmanager
from typing import cast
from mlir_ktir.ir import *

from ._mlir_libs._ktir import register_dialects


@contextmanager
def scheduler_context():
    reg = DialectRegistry()
    register_dialects(reg)

    with Context() as ctx:
        # MLIR does not properly define the signature of __enter__...
        ctx = cast(Context, ctx)
        ctx.append_dialect_registry(reg)
        yield ctx
