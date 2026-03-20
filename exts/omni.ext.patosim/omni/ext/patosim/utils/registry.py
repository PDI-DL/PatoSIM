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


from collections import OrderedDict
from typing import TypeVar, Generic

T = TypeVar('T')

class Registry(Generic[T]):

    def __init__(self):
        self.items = OrderedDict()

    def register(self):
        def _register(cls):
            self.items[cls.__name__] = cls
            return cls
        return _register

    def names(self):
        return self.items.keys()
    
    def get(self, name: str) -> T:
        return self.items[name]
    
    def get_index(self, index: int) -> T:
        return list(self.items.values())[index]
    
    