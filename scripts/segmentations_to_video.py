import cv2
import numpy as np
from pathlib import Path
import argparse

def create_fixed_colormap():
    """Create a fixed colormap of 16 distinct pastel colors."""
    colors = [
        [255, 182, 193],  # Light pink
        [176, 224, 230],  # Powder blue
        [255, 218, 185],  # Peach
        [221, 160, 221],  # Plum
        [176, 196, 222],  # Light steel blue
        [152, 251, 152],  # Pale green
        [255, 255, 224],  # Light yellow
        [230, 230, 250],  # Lavender
        [255, 228, 225],  # Misty rose
        [240, 255, 240],  # Honeydew
        [255, 240, 245],  # Lavender blush
        [224, 255, 255],  # Light cyan
        [250, 235, 215],  # Antique white
        [245, 255, 250],  # Mint cream
        [255, 228, 196],  # Bisque
        [240, 248, 255],  # Alice blue
    ]
    return np.array(colors, dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Convert segmentation PNGs to colorized video')
    parser.add_argument('input_dir', type=str, help='Directory containing segmentation PNGs')
    parser.add_argument('output_path', type=str, help='Output video path (e.g., output.mp4)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--normals_dir', type=str, help='Directory containing surface normal .npy files')
    parser.add_argument('--depth_dir', type=str, help='Directory containing 16-bit inverse depth PNG files')
    args = parser.parse_args()

    # Get sorted list of PNG files
    input_dir = Path(args.input_dir)
    png_files = sorted(list(input_dir.glob('*.png')))
    
    if not png_files:
        raise ValueError(f"No PNG files found in {input_dir}")

    # Get corresponding normal files if provided
    normals_dir = None
    if args.normals_dir:
        normals_dir = Path(args.normals_dir)
        
    # Get depth directory if provided
    depth_dir = None
    if args.depth_dir:
        depth_dir = Path(args.depth_dir)

    # Read first image to get dimensions
    first_img = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
    height, width = first_img.shape

    # Create fixed colormap
    colormap = create_fixed_colormap()
    n_colors = len(colormap)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))

    # Process each frame
    for png_file in png_files:
        # Read segmentation image
        seg = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
        
        # Apply median filter to reduce noise in segmentation
        # seg = cv2.medianBlur(seg, 5)  # kernel size of 5, adjust if needed
        
        # Load corresponding normal map if available
        normal_map = None
        if normals_dir:
            normal_file = normals_dir / (png_file.stem + '.npy')
            if normal_file.exists():
                normal_map = np.load(str(normal_file))
        
        # Load corresponding depth map if available
        depth_map = None
        if depth_dir:
            depth_file = depth_dir / png_file.name
            if depth_file.exists():
                depth_map = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                depth_map = depth_map / 65535.0
            else:
                print(f"Depth file {depth_file} does not exist")

        # Create base colored frame
        colored_frame = colormap[seg % len(colormap)]
        
        # Apply shading based on normal map and depth map
        if normal_map is not None or depth_map is not None:
            # Initialize combined shading
            shading = np.ones((height, width), dtype=float)
            
            # Apply normal-based shading if available
            if normal_map is not None:
                normal_xyz = 2.0 * normal_map[..., :3] - 1.0
                light_dir = np.array([-0.5, 0.5, 1])
                light_dir = light_dir / np.sqrt(np.sum(light_dir**2))
                normal_shading = np.clip(np.dot(normal_xyz, light_dir), 0, 1)
                normal_shading /= normal_map[..., 3]
                normal_shading = 0.5 + 0.5 * normal_shading
                
                shading *= normal_shading

            # Apply depth-based shading if available
            if depth_map is not None:
                # shading *= (depth_map**0.1)
                pass

            # Apply combined shading to colored frame
            shading = shading.reshape(height, width, 1)
            colored_frame = (colored_frame * shading).astype(np.uint8)

        # Write frame to video
        out.write(colored_frame)

    # Release video writer
    out.release()
    print(f"Video saved to {args.output_path}")

if __name__ == "__main__":
    main()
