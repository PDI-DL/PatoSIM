# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension


ext_modules = [
    Pybind11Extension(
        "mobility_gen_path_planner._mobility_gen_path_planner_C",
        ["src/python_bindings.cpp"],  # Sort source files for reproducibility
        include_dirs=["src"]
    ),
]

setup(
    name="mobility_gen_path_planner",
    packages=find_packages(),
    ext_modules=ext_modules
)