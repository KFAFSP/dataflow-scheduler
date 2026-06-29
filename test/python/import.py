# RUN: python %s

# This is a trivial test that checks whether all the packages we expected to
# produce can be imported.

import mlir_scheduler
import mlir_scheduler.ir

import mlir_scheduler.dialects.arith

import mlir_scheduler.dialects.ktdp

import mlir_scheduler.dialects.ktdf
