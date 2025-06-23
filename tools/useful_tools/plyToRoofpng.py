import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from PIL import Image

# Read PLY file
def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

# Find the lowest point and subtract its height from all points
def adjust_height(points):
    min_z = np.min(points[:, 2])
    points[:, 2] -= min_z
    return points

# Create 128x128 grid and find the highest point in each grid
def create_grid(points, grid_size=128):
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Calculate scale for x and y to keep aspect ratio
    range_x = max_x - min_x
    range_y = max_y - min_y
    max_range = max(range_x, range_y)
    
    # Redefine grid boundaries to keep aspect ratio
    x_step = max_range / grid_size
    y_step = max_range / grid_size

    # Use numpy broadcasting and condition filtering for optimization
    grid_points = np.floor((points[:, :2] - [min_x, min_y]) / [x_step, y_step]).astype(int)

    grid_z = np.zeros((grid_size, grid_size)) - np.inf
    for idx, (x_idx, y_idx) in enumerate(grid_points):
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            grid_z[x_idx, y_idx] = max(grid_z[x_idx, y_idx], points[idx, 2])
    return grid_z
    
# Use DBSCAN to separate high points and remove ground points

def filter_high_points(points, eps=0.27, min_samples=6, z_threshold=3.0):
    # Only cluster points with Z above threshold
    high_points = points[points[:, 2] > z_threshold]

    if len(high_points) == 0:
        return points
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(high_points[:, :2])
    labels = clustering.labels_
    
    # Keep points in valid clusters, remove ground points
    return high_points

# Filter out points below min_z + 3 meters

def filter_low_points(points, z_threshold=3.0):
    # Find the lowest point
    min_z = np.min(points[:, 2])
    
    # Points below ground height + 3 meters are set to -inf
    filtered_points = points.copy()
    filtered_points[filtered_points[:, 2] < min_z + z_threshold, 2] = -np.inf
    return filtered_points

# Apply formula to recalculate height and generate top view PNG

def point_cloud_to_height_map(points_f, output_file,randomfilter, grid_size=256, Z=15, ):
    # Use formula to calculate new height
    # points[:, 2] = (2 / Z) * (points[:, 2] - 0.5 * (z_max + z_min))
    # Height * 256 to remap height value
    points=points_f.copy()
    points[:,2]=(points[:,2]*256)

    # Calculate x and y range
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Calculate x and y range, use larger range to keep aspect ratio
    range_x = max_x - min_x
    range_y = max_y - min_y
    max_range = max(range_x, range_y)

    # Calculate offset to center point cloud
    offset_x = (max_range - range_x) / 2
    offset_y = (max_range - range_y) / 2

    # Create array for pixel heights, initialize to minimum value
    pixel_height_map = np.full((grid_size, grid_size), -np.inf)

    # Redefine grid boundaries to keep aspect ratio
    x_step = max_range / (grid_size-2)
    y_step = max_range / (grid_size-2)

    for i in range(grid_size-2):
        for j in range(grid_size-2):
            # Adjust boundary condition: left closed, right open; last row/col special case
            x_min = min_x + i * x_step - offset_x
            x_max = min_x + (i + 1) * x_step - offset_x if i < grid_size - 1 else min_x + grid_size * x_step - offset_x
            y_min = min_y + j * y_step - offset_y
            y_max = min_y + (j + 1) * y_step - offset_y if j < grid_size - 1 else min_y + grid_size * y_step - offset_y

            # Filter points in this grid
            grid_points = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                                 (points[:, 1] >= y_min) & (points[:, 1] <= y_max)]
            
            if len(grid_points) > 0:
                # Find the highest point in each grid
                highest_point = grid_points[np.argmax(grid_points[:, 2])]
                pixel_height_map[i+1,j+1]=highest_point[2]

    # Randomly set some pixels to 0
    mask = np.random.choice([True, False], size=pixel_height_map.shape, p=[1-randomfilter, randomfilter])
    pixel_height_map[mask] = 0

    # Replace -inf with 0
    pixel_height_map[pixel_height_map == -np.inf] = 0

    pixel_height_map = pixel_height_map.astype(np.uint16)
    print("range of z:",np.min(pixel_height_map),np.max(pixel_height_map))

    # Create and save image (I indicates 32-bit integer pixels)
    img = Image.fromarray(pixel_height_map, mode='I;16')
    img=img.rotate(90, expand=True)
    img.save(output_file)
    print(f"Saved top view image as {output_file}")

# Main program: iterate all PLY files in folder

def process_ply_files_in_folder(folder_path):
    # Iterate all files in folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.ply'):
            file_path = os.path.join(folder_path, file_name)
            try:
                print(f"Processing file: {file_path}")
                # Read PLY file and process
                points = load_ply(file_path)
                # Adjust height
                # points = adjust_height(points)
                # Create grid and find highest point
                # points = create_grid(points, grid_size=128)
                # Use DBSCAN to filter high points
                # filtered_points = filter_high_points(highest_points)
                # Filter low points
                filtered_points = filter_low_points(points)
                # Recalculate height and output as top view PNG
                output_file = os.path.join(folder_path + "\png", f"{file_name}_topview.png")
                point_cloud_to_height_map(filtered_points, output_file, randomfilter = 0.15)
                output_file_full = os.path.join(folder_path + "\png", f"{file_name}_topview_full.png")
                point_cloud_to_height_map(filtered_points, output_file_full, randomfilter = 1)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Example call
folder_path = r'D:\thesis\3dbag\ft-point\roof'  # Replace with actual folder path
process_ply_files_in_folder(folder_path)
