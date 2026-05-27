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

import os
import lit.formats
import lit.llvm
from lit.llvm.subst import ToolSubst

llvm_config = lit.llvm.llvm_config

config.name = "KTIR-scheduler"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir", ".py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.ktir_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

tools = ["scheduler", "FileCheck", "not"]
tool_dirs = [config.ktir_tools_dir, config.llvm_tools_dir]

if config.enable_python_bindings:
    config.environment["PYTHONPATH"] = os.getenv("MLIR_LIT_PYTHONPATH", "")
    llvm_config.with_environment(
        "PYTHONPATH",
        [
            os.path.join(config.ktir_obj_root, "python_packages", "ktir"),
        ],
        append_path=True,
    )
    tools.extend([ToolSubst("python", config.python_executable)])

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.substitutions.append(("%ktir_test_root", config.ktir_test_root))

# Quote tool paths that contain spaces so shell commands work correctly
config.substitutions = [
    (key, '"%s"' % val if val and " " in val and not val.startswith('"') else val)
    for key, val in config.substitutions
]
