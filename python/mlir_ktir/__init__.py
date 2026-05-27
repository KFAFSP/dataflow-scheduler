from contextlib import contextmanager
from mlir_ktir.ir import *

from ._mlir_libs._ktir import register_dialects

@contextmanager
def scheduler_context():
    reg = DialectRegistry()
    register_dialects(reg)

    with Context() as ctx:
        ctx.append_dialect_registry(reg)
        yield ctx
