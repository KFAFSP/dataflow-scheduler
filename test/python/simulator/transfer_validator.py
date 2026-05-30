# RUN: python %s %ktir_test_root/Transform/PathExpansion/basic.mlir 2>&1 | FileCheck %s

import sys

from mlir_ktir import scheduler_context
import mlir_ktir.ir as ir
from mlir_ktir.scheduler import ArchitectureGraph
from mlir_ktir.simulator.diagnostic import DIAGNOSTIC_LEVEL, STDERR, Severity
from mlir_ktir.simulator.validator import TransferValidator

input_file = sys.argv[1]

spyre_arch = ArchitectureGraph.createBuiltInSpyre()

with scheduler_context():
    module = ir.Module.parseFile(input_file)

    with DIAGNOSTIC_LEVEL.set(Severity.WARNING):
        STDERR.options.overflow = "ignore"
        result = TransferValidator(spyre_arch).run(module.operation)

# CHECK: 123:{{[0-9]+}}: error: no direct path from HBM to SFP
# CHECK-COUNT-2: |
# CHECK-DAG: 123:{{[0-9]+}}: error: no direct path from HBM to SFP
# CHECK-COUNT-2: |
# CHECK-DAG: 124:{{[0-9]+}}: error: no direct path from SFP to HBM
# CHECK-COUNT-2: |
# CHECK-DAG: 131:{{[0-9]+}}: error: no direct path from HBM to SFP
# CHECK-COUNT-2: |
# CHECK-DAG: 132:{{[0-9]+}}: error: no direct path from HBM to SFP
# CHECK-COUNT-2: |
# CHECK-DAG: 146:{{[0-9]+}}: error: no direct path from SFP to HBM
# CHECK-COUNT-2: |
# CHECK-EMPTY:
