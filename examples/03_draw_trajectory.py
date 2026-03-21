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

import argparse
import matplotlib.pyplot as plt
import numpy as np
from reader import Reader


parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
args = parser.parse_args()

reader = Reader(recording_path=args.recording_path)

occupancy_map = reader.read_occupancy_map()

points_world = []

for index in range(len(reader)):
    state_dict = reader.read_state_dict_common(index)

    position = state_dict['robot.position']

    points_world.append(position[0:2])

points_world = np.array(points_world)

points_image = occupancy_map.world_to_pixel_numpy(points_world)

plt.imshow(occupancy_map.freespace_mask(), cmap="gray")
plt.plot(points_image[:, 0], points_image[:, 1], 'g-')
plt.show()
