# Description: This script reads a PLY file and projects the mesh onto a 128x128 grid in the XY plane.
# This script can only process one ply file at a time.

import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Read PLY file
# mesh = trimesh.load_mesh(r"D:\thesis\3dbag\ft and point\footprint\0606100000001709_ft.ply")
mesh = trimesh.load_mesh(r"D:\thesis\BuildingPCC_NL_50k\selected\mesh\bag_0518100000336345.ply")

# Assume the grid range and resolution
xmin, xmax = mesh.bounds[0][0], mesh.bounds[1][0]
ymin, ymax = mesh.bounds[0][1], mesh.bounds[1][1]
resolution = 128

# Calculate xy range
range_x = xmax - xmin
range_y = ymax - ymin
max_range = max(range_x, range_y)

# Map xy coordinates to 128x128 grid
x_normalized = np.linspace(xmin, xmax, resolution)
y_normalized = np.linspace(ymin, ymax, resolution)

# Create a 128x128 z value matrix, initialize to -inf to indicate empty
z_matrix = np.full((resolution, resolution), -np.inf)

# For each pixel, cast a ray and find intersection with mesh
for ix in range(resolution):
    for iy in range(resolution):
        # Construct a ray in XY plane, assume z=1000 (or much higher than mesh height)
        ray_origin = np.array([x_normalized[ix], y_normalized[iy], 100000])
        ray_direction = np.array([0, 0, -1])  # Shoot down along z axis

        # Find intersection of ray and mesh
        locations, index_ray = mesh.ray.intersects_location(ray_origins=ray_origin, ray_directions=ray_direction)

        # If there is intersection, take the closest z value
        if locations.size > 0:
            z_matrix[iy, ix] = np.max(locations[:, 2])  # Take the maximum z value

# Output z matrix
print("Z Matrix (128x128 grid of max Z values):")
print(z_matrix)

# Visualize z matrix
plt.imshow(z_matrix, cmap='viridis', origin='lower')
plt.colorbar(label="Z value")
plt.title("Max Z values projection on XY plane")
plt.show()

