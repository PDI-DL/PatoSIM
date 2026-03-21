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
from reader import Reader


parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--normals_enabled", type=bool, default=False)
args = parser.parse_args()

reader = Reader(recording_path=args.recording_path)

images = reader.read_state_dict(index=0)

nrows = 3
if args.normals_enabled:
    nrows += 1

plt.subplot(nrows, 2, 1)
plt.title('left rgb')
plt.imshow(images['robot.front_camera.left.rgb_image'])
plt.subplot(nrows, 2, 2)
plt.title('right rgb')
plt.imshow(images['robot.front_camera.right.rgb_image'])
plt.subplot(nrows, 2, 3)
plt.title('left depth (inverse)')
plt.imshow(1.0 / images['robot.front_camera.left.depth_image'])
plt.subplot(nrows, 2, 4)
plt.title('right depth (inverse)')
plt.imshow(1.0 / images['robot.front_camera.right.depth_image'])
plt.subplot(nrows, 2, 5)
plt.title('left instance id')
plt.imshow(images['robot.front_camera.left.instance_id_segmentation_image'])
plt.subplot(nrows, 2, 6)
plt.title('right instance id')
plt.imshow(images['robot.front_camera.right.instance_id_segmentation_image'])

if args.normals_enabled:
    plt.subplot(nrows, 2, 7)
    plt.title('left instance id')
    plt.imshow(images['robot.front_camera.left.normals_image'])
    plt.subplot(nrows, 2, 8)
    plt.title('right instance id')
    plt.imshow(images['robot.front_camera.right.normals_image'])
plt.show()
    