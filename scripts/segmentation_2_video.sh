python segmentations_colored_video.py \
  "$HOME/MobilityGenData/replays/2025-09-09T12:12:16.390205/state/segmentation/robot.front_camera.left.segmentation_image" \
  "output.mp4" \
  --fps 15 \
  --depth_dir "$HOME/MobilityGenData/replays/2025-09-09T12:12:16.390205/state/depth/robot.front_camera.left.depth_image" \
  --output_root "$HOME/MobilityGenData/replays/2025-09-09T12:12:16.390205" \
  --save_depth_vis
