# RUN: python %s %ktir_test_root/Dialects/KTDF/pipeline-stage-ops.mlir

import sys

input_file_name = sys.argv[1]

from mlir_ktir import scheduler_context
from mlir_ktir.ir import Module, Operation, WalkOrder, WalkResult

with scheduler_context() as ctx:
    ctx.allow_unregistered_dialects = True
    mod = Module.parseFile(input_file_name)

    def walk_fn(op: Operation) -> WalkResult:
        print(op)
        return WalkResult.ADVANCE

    mod.operation.walk(walk_fn, WalkOrder.POST_ORDER)
