# RUN: python %s

from mlir_ktir import scheduler_context
import mlir_ktir.dialects.ktdf as ktdf

with scheduler_context() as ctx:
    ctx.load_all_available_dialects()

    # TokenType
    token_type = ktdf.TokenType.get()
    assert token_type.__str__() == '!ktdf.token'

    # FifoSlotType
    fifo_slot_type = ktdf.FifoSlotType.get(ktdf.FifoType.SfpToHbm, 42, token_type)
    assert fifo_slot_type.__str__() == '!ktdf.fifo.slot<"sfp-to-hbm", 42x!ktdf.token>'
    assert fifo_slot_type.fifoType == ktdf.FifoType.SfpToHbm
    assert fifo_slot_type.numElements == 42
    assert fifo_slot_type.elementType == token_type

    # MemorySpaceAttr
    memory_space = ktdf.MemorySpaceAttr.get(ktdf.MemorySpace.HBM)
    assert memory_space.__str__() == '#ktdf.memory_space<HBM>'
    assert memory_space.value == ktdf.MemorySpace.HBM

    # LoopTypeAttr
    loop_type = ktdf.LoopTypeAttr.get(ktdf.LoopType.ParallelLoop)
    assert loop_type.__str__() == '#ktdf.loop_type<parallel_loop>'
    assert loop_type.value == ktdf.LoopType.ParallelLoop

    # ApplicableUnitsAttr
    applicable_units = ktdf.ApplicableUnitsAttr.get()
    assert applicable_units.__str__() == '#ktdf.applicable_units<[]>'
    assert applicable_units.value == []

    component_types = [ktdf.ComponentType.L3LU, ktdf.ComponentType.L3SU]
    applicable_units = ktdf.ApplicableUnitsAttr.get(component_types)
    assert applicable_units.__str__() == '#ktdf.applicable_units<[L3LU, L3SU]>'
    assert applicable_units.value == component_types
