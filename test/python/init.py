# RUN: python %s

import mlir_ktir
from mlir_ktir.ir import Context, DialectRegistry

reg = DialectRegistry()
mlir_ktir.register_dialects(reg)

ctx = Context()
ctx.append_dialect_registry(reg)
ctx.load_all_available_dialects()
