import open3d as o3d
import laspy
import numpy as np
import os
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt

import laspy
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree

def convert_las_to_ply(las_file_path, ply_file_path, color=False):
    # read las file
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()  # extract x, y, z coordinates
    # # Print point cloud column names
    # colums=las.point_format.dimension_names
    # for col in colums:
    #     print(col)
    
    if color:
        if 'red' in las.point_format and 'green' in las.point_format and 'blue' in las.point_format:
            colors = np.vstack((las.red, las.green, las.blue)).transpose()
            # Normalize color to [0, 1]
            colors = colors / 65535.0 if colors.max() > 1 else colors
        else:
            colors = None  # If no color, default to None

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # if colors is not None:
    #     point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save as PLY file
    o3d.io.write_point_cloud(ply_file_path, point_cloud)
    # print(f"Conversion complete: {ply_file_path}")



def point_cloud_to_height_map(points, range_bbx=None,grid_size=256, pixel_size=None):
    # Rescale z-coordinate
    points[:, 2] *= 257  # 257 is a magic number to rescale z-coordinate to uint16

    # print(max(points[:, 2]))
    # Compute grid parameters
    if range_bbx is None:
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        max_range = max(max_x - min_x, max_y - min_y)
    else:
        max_range = range_bbx[0]
        min_x, max_x = range_bbx[1], range_bbx[2]
        min_y, max_y = range_bbx[3], range_bbx[4]
    # min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    # min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    if pixel_size:
        max_range = pixel_size * grid_size

    offset_x = (max_range - (max_x - min_x)) / 2
    offset_y = (max_range - (max_y - min_y)) / 2


    # Use binned_statistic_2d for faster binning and max computation
    stat, _, _, _ = binned_statistic_2d( # Assign data points to 2D grid and compute max value in each grid
        points[:, 0], points[:, 1], points[:, 2],
        statistic='max', bins=grid_size,
        range=[[min_x - offset_x, min_x - offset_x + max_range],
               [min_y - offset_y, min_y - offset_y + max_range]]
    )

    # Replace NaN (no points in bin) with 0 and convert to uint16
    # stat = np.nan_to_num(stat, nan=0).astype(np.uint16)
    stat = np.nan_to_num(stat, nan=0, neginf=0).astype(np.uint16)
    
    return stat.T

def convert_las_to_roof(las_file_path,resolution,range_bbx=None,z_threshold=0, color=False, pixel_size=None,min_z=None):
    '''
    Convert a LAS file to a height map
    :param las_file_path: path to the LAS file
    :param resolution: resolution of the height map
    :param max_range: max range of the height map(default: None, use the max range of the point cloud)
                    If max_range is specified, the height map will be square with the side length of max_range
    :param z_threshold: threshold for filtering points
    :return: height map
    '''
    # read las file
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()  # extract x, y, z coordinates

    if min_z is None:
        min_z = np.zeros_like(points[:, 2])
    # print(max(points[:, 2]),"1st step")
    
    # Points below z_threshold meters are set to -inf
    points[:, 2] = points[:, 2] - min_z
    filtered_points = points.copy()
    # filtered_points[filtered_points[:, 2] < min_z + z_threshold, 2] = -np.inf
    filtered_points[filtered_points[:, 2] < z_threshold, 2] = 0


    # Convert point cloud to height map
    height_map = point_cloud_to_height_map(filtered_points,range_bbx, grid_size=resolution, pixel_size=pixel_size)
    return height_map

def cal_point_mean_dis(laspath):


    # Read LAS file
    las = laspy.read(laspath)

    # Get point cloud coordinates
    points = np.vstack((las.x, las.y, las.z)).T #v

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(points)
    
    # Query the two nearest points for each point (k=2, exclude itself)
    distances, _ = tree.query(points, k=2)  # First column is itself (distance 0), second column is nearest neighbor
    
    # Extract nearest neighbor distances (excluding itself)
    nearest_distances = distances[:, 1]
    
    # Calculate mean distance
    mean_distance = np.mean(nearest_distances)
    
    return mean_distance
