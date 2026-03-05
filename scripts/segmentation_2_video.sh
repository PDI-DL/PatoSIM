INPUT_DIR_SEGMENTATION="/media/pdi_4/external/mobilitygen/2025-09-09T15:32:06.175162/state/segmentation/robot.front_camera.left.instance_id_segmentation_image"
INPUT_DIR_DEPTH="/media/pdi_4/external/mobilitygen/2025-09-09T15:32:06.175162/state/depth/robot.front_camera.left.depth_image"
OUTPUT_DIR="/media/pdi_4/external/mobilitygen/2025-09-09T15:32:06.175162/state/segmentation"

python segmentations_colored_video.py \
    "$INPUT_DIR_SEGMENTATION" \
    "$OUTPUT_DIR" \
    --fps 30 \
    --depth_dir "$INPUT_DIR_DEPTH" \
    --video_name "output_video.mp4" \
    --save_depth_vis
