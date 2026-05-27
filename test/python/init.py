# RUN: python %s

from mlir_ktir import scheduler_context

with scheduler_context() as ctx:
    ctx.load_all_available_dialects()
