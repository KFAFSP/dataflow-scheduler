# RUN: python %s %ktir_test_root/Transform/PathExpansion/basic.mlir

import sys

from mlir_ktir import scheduler_context
import mlir_ktir.ir as ir
from mlir_ktir.scheduler import ArchitectureGraph
from mlir_ktir.simulator.diagnostic import (
    DiagnosticVerifier,
    DIAGNOSTIC_LEVEL,
    DIAGNOSTIC_LEVEL_QUIET,
    Severity,
)
from mlir_ktir.simulator.validator import TransferValidator

input_file = sys.argv[1]

spyre_arch = ArchitectureGraph.createBuiltInSpyre()

with scheduler_context():
    module = ir.Module.parseFile(input_file)

    with DIAGNOSTIC_LEVEL.set(DIAGNOSTIC_LEVEL_QUIET):
        result = TransferValidator(spyre_arch).run(module.operation)

    DiagnosticVerifier(
        (Severity.ERROR, 123, "no transfer path from HBM to SFP"),
        (Severity.ERROR, 123, "no transfer path from HBM to SFP"),
        (Severity.ERROR, 124, "no transfer path from SFP to HBM"),
        (Severity.ERROR, 131, "no transfer path from HBM to SFP"),
        (Severity.ERROR, 132, "no transfer path from HBM to SFP"),
        (Severity.ERROR, 146, "no transfer path from SFP to HBM"),
    ).verify(result, abort_on_fail=True)
