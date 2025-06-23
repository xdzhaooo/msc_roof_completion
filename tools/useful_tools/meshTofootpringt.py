# This code is used to project 3D mesh to 2D footprint image

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from PIL import Image
import os
import concurrent.futures
import cv2

def project_mesh_to_xy(mesh_file, resolution=128):
    # Load PLY mesh
    mesh = trimesh.load(mesh_file)
    print("mesh faces",len(mesh.faces))

    # Get bounding box
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    min_x, min_y = bounds[0][:2]
    max_x, max_y = bounds[1][:2]
    
    # Define grid resolution and pixel size
    x_range = max_x - min_x
    y_range = max_y - min_y
    max_range=max(x_range,y_range)
    pixel_size_x = max_range/resolution
    pixel_size_y = max_range/resolution
    pixel_size = max_range / resolution
    # offset
    offset_x = (max_range - x_range) / 2
    offset_y = (max_range - y_range) / 2

    # Initialize projection image
    projection = np.zeros((resolution, resolution), dtype=np.uint8)
    vertices_2d = (mesh.vertices[:, :2] - [min_x - offset_x, min_y - offset_y]) / pixel_size
    vertices_2d = np.clip(vertices_2d, 0, resolution - 1)
    
    count = 0

    # # Project mesh to xy plane (alternative method)
    # for face in mesh.faces:
    #     count += 1
    #     if count % 100 == 0:
    #         print(f"Processing face {count}...")
    #     # Get triangle vertices
    #     vertices = mesh.vertices[face][:, :2]  # Only keep x, y coordinates
    #     # Map vertices from real coordinates to pixel coordinates
    #     pixels = (vertices - [min_x-offset_x, min_y-offset_y]) / [pixel_size_x, pixel_size_y]
    #     pixels = np.clip(pixels, 0, resolution - 1)
    #     # Create a polygon object
    #     polygon = Polygon(pixels)
    #     # Rasterize triangle and fill in projection image
    #     if polygon.is_valid:
    #         # Get all grid coordinates of the polygon
    #         for x in range(resolution):
    #             for y in range(resolution):
    #                 if polygon.contains(Point(x, y)):
    #                     projection[x, y] = 1

    for face in mesh.faces:
        pixels = np.round(vertices_2d[face]).astype(np.int32)
        cv2.fillPoly(projection, [pixels], 1)
    

    return projection[::-1]  # Flip image to match matplotlib coordinate system

def write_to_png(image, output_file):
    image = Image.fromarray(image, mode='L')
    image.save(output_file)
    print(f"Saved projection image as {output_file}")

# Main program: read all PLY files in folder and project

def project_mesh_to_xy_folder(folder_path):
    # Iterate all PLY files in folder
    for file in os.listdir(folder_path):
        if file.endswith(".ply"):
            ply_file = os.path.join(folder_path, file)
            projection_image = project_mesh_to_xy(ply_file)
            output_file = os.path.join(folder_path, f"{file[:-4]}.png")
            write_to_png(projection_image, output_file)

# Test
path=r"D:\thesis\3dbag\ft-point\footprint"
project_mesh_to_xy_folder(path)
