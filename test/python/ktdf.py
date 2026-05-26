# RUN: python %s

import mlir_ktir
from mlir_ktir.ir import Context, DialectRegistry

reg = DialectRegistry()
mlir_ktir.register_dialects(reg)

ctx = Context()
ctx.append_dialect_registry(reg)
ctx.load_all_available_dialects()

import mlir_ktir.dialects.ktdf as ktdf

# TokenType
token_type = ktdf.TokenType.get(context=ctx)
assert token_type.__str__() == '!ktdf.token'

# FifoSlotType
fifo_slot_type = ktdf.FifoSlotType.get(ktdf.FifoType.SfpToHbm, 42, token_type)
assert fifo_slot_type.__str__() == '!ktdf.fifo.slot<"sfp-to-hbm", 42x!ktdf.token>'
assert fifo_slot_type.fifoType == ktdf.FifoType.SfpToHbm
assert fifo_slot_type.numElements == 42
assert fifo_slot_type.elementType == token_type

# ApplicableUnitsAttr
applicable_units = ktdf.ApplicableUnitsAttr.get(context=ctx)
assert applicable_units.__str__() == '#ktdf.applicable_units<[]>'
assert applicable_units.componentTypes == []

component_types = [ktdf.ComponentType.L3LU, ktdf.ComponentType.L3SU]
applicable_units = ktdf.ApplicableUnitsAttr.get(component_types, context=ctx)
assert applicable_units.__str__() == '#ktdf.applicable_units<[L3LU, L3SU]>'
assert applicable_units.componentTypes == component_types
