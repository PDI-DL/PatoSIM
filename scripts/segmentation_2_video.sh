python segmentations_colored_video.py \
  "$HOME/MobilityGenData/replays/2025-08-19T16:09:59.275637/state/segmentation/robot.front_camera.left.segmentation_image" \
  "output.mp4" \
  --fps 15 \
  --depth_dir "$HOME/MobilityGenData/replays/2025-08-19T16:09:59.275637/state/depth/robot.front_camera.left.depth_image" \
  --output_root "$HOME/MobilityGenData/replays/2025-08-19T16:09:59.275637" \
  --save_depth_vis
