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


import cv2
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert image sequence to video')
    parser.add_argument('input_dir', type=str, help='Directory containing image files')
    parser.add_argument('output_path', type=str, help='Output video path (e.g., output.mp4)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg'], 
                       help='Image format to process (png or jpg)')
    args = parser.parse_args()

    # Get sorted list of image files
    input_dir = Path(args.input_dir)
    if args.format.lower() == 'jpg':
        image_files = sorted(list(input_dir.glob('*.jpg')))
    else:
        image_files = sorted(list(input_dir.glob('*.png')))
    
    if not image_files:
        raise ValueError(f"No {args.format.upper()} files found in {input_dir}")

    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    height, width = first_img.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))

    # Process each frame
    for image_file in image_files:
        # Read image
        frame = cv2.imread(str(image_file))
        
        # Write frame to video
        out.write(frame)

    # Release video writer
    out.release()
    print(f"Video saved to {args.output_path}")

if __name__ == "__main__":
    main()