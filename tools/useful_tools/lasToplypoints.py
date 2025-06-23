import open3d as o3d
import laspy
import numpy as np
import os
import plyToRoofpng as prp

def convert_las_to_ply(las_file_path, ply_file_path):
    # Read LAS file
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()  # Extract point cloud coordinates
    
    # If there is color information
    # if 'red' in las.point_format and 'green' in las.point_format and 'blue' in las.point_format:
    #     colors = np.vstack((las.red, las.green, las.blue)).transpose()
    #     # Normalize color to [0, 1]
    #     colors = colors / 65535.0 if colors.max() > 1 else colors
    # else:
    #     colors = None  # If no color, default to None

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # if colors is not None:
    #     point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save as PLY file
    o3d.io.write_point_cloud(ply_file_path, point_cloud)
    print(f"Conversion complete: {ply_file_path}")

def processing(input_file, output_file):
    convert_las_to_ply(input_file, output_file)

input_path = "las/"

for file in os.listdir(input_path):
    if os.path.isdir(input_path + file):
        for subfile in os.listdir(input_path + file):
            if subfile.endswith(".las"):
                input_file = input_path + file + "/" + subfile
                output_file = input_path + "/ply" + "/" + file + "_"  + subfile.replace(".las", ".ply")
                processing(input_file, output_file)