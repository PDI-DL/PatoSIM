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


import PIL.Image
import time
import numpy as np
import matplotlib.pyplot as plt

from mobility_gen_path_planner import generate_paths


image = PIL.Image.open("cube_map.png")

freespace = (np.asarray(image) > 0).astype(np.uint8)

# freespace = np.concatenate([freespace, freespace, freespace], axis=0)
# freespace = np.concatenate([freespace, freespace, freespace], axis=1)

count = 10
t0 = time.perf_counter()
for i in range(count):
    output = generate_paths((128, 128), freespace)
t1 = time.perf_counter()

print("Latency (ms):")
print(1000. * (t1 - t0) / count)

# print(output)


plt.imshow(output.distance_to_start)


colors = 'rgbkcym'

for i in range(10):
    path = output.sample_random_path()

    plt.plot(path[:, 1], path[:, 0], colors[i % len(colors)] + '.-')
plt.show()