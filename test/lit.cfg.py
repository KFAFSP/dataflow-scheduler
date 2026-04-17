#
# Copyright 2026 The KTIR Scheduler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Taken from mlir(https://github.com/llvm/llvm-project/blob/main/mlir/test) and modified as required
# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'KTIR-Scheduler::MLIR'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.toy', '.ll', '.tc', '.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(("%mlir_src_root", config.mlir_src_root))
llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt',
                   'lit.cfg.py', 'lit.site.cfg.py']
config.available_features.add(config.build_type)
# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# For code coverage
llvm_config.with_environment('LLVM_PROFILE_FILE', os.path.join(config.scheduler_obj_root, 'profdata/scheduler_lit_tests.%p.profraw'), append_path=False)
tool_dirs = [config.llvm_tools_dir, config.scheduler_tools_dir]
tools = [
    'scheduler'
]


# The following tools are optional
tools.extend([
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
])
llvm_config.add_tool_substitutions(tools, tool_dirs)


# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment['FILECHECK_OPTS'] = "-enable-var-scope --allow-unused-prefixes=false"


# LLVM can be configured with an empty default triple
# by passing ` -DLLVM_DEFAULT_TARGET_TRIPLE="" `.
# This is how LLVM filters tests that require the host target
# to be available for JIT tests.
if config.target_triple:
    config.available_features.add('default_triple')

# Made with Bob
