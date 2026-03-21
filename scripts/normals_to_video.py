import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def normalize_normals(normal_map):
    """Normalize the normal vectors to range [-1, 1]"""
    return (normal_map - np.min(normal_map)) / (np.max(normal_map) - np.min(normal_map)) * 2 - 1

def visualize_normals(normal_map):
    """Convert normal map to RGB visualization
    Normal map is expected to be HxWx4 with format (X,Y,Z,W)
    Returns HxWx3 RGB image with values in [0, 255]
    """
    # Extract XYZ components and normalize
    xyz = normal_map[..., :3]
    
    # Convert from [-1,1] to [0,1] range
    rgb = (xyz + 1.0) / 2.0
    
    # Convert to uint8 [0,255]
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert normal map .npy files to video visualization')
    parser.add_argument('input_dir', type=str, help='Directory containing .npy files')
    parser.add_argument('output_path', type=str, help='Output video file path')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second (default: 30)')
    
    args = parser.parse_args()
    
    # Get all .npy files sorted
    npy_files = sorted(list(Path(args.input_dir).glob("*.npy")))
    if not npy_files:
        print(f"No .npy files found in {args.input_dir}!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read first file to get dimensions
    first_frame = np.load(npy_files[0])
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))
    
    # Process each frame
    print("Processing frames...")
    for npy_file in tqdm(npy_files):
        # Load normal map
        normal_map = np.load(npy_file)
        
        # Ensure the normal map is normalized
        normal_map = normalize_normals(normal_map)
        
        # Convert to RGB visualization
        rgb_frame = visualize_normals(normal_map)
        
        # OpenCV uses BGR format
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        out.write(bgr_frame)
    
    # Release video writer
    out.release()
    print(f"Video saved as {args.output_path}")

if __name__ == "__main__":
    main()