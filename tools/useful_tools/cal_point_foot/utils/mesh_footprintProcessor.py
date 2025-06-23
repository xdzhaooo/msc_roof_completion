import trimesh
import numpy as np
import cv2
import matplotlib.pyplot as plt

def project_mesh_to_xy(mesh_file, resolution=128, pixel_size=None):
    '''
    mesh_file: ply file path
    resolution: resolution of the output image
    return: binary numpy array(1 for footprint, 0 for background), number of faces
    '''
    # read ply file
    mesh = trimesh.load_mesh(mesh_file)
    mesh_faces = len(mesh.faces)
    slop_area_ratio, slope_face_num = slop_roof_detector(mesh)

    # get the bounding box
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    min_z = bounds[0][2] 
    min_x, min_y = bounds[0][:2]
    max_x, max_y = bounds[1][:2]
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    max_range = max(diff_x, diff_y)
    # define the resolution of the grid and the size of each pixel
    x_range = max_x - min_x
    y_range = max_y - min_y
    max_range = max(x_range, y_range) # the maximum range of the bounding box
    if pixel_size is None:
        pixel_size = max_range / resolution
    else:
        max_range = pixel_size * resolution
    offset_x = (max_range - x_range) / 2
    offset_y = (max_range - y_range) / 2

    # check the highest point of the mesh
    max_z = np.max(mesh.vertices[:, 2])
    resolution = int(resolution)
    # initialize the projection image
    projection = np.zeros((resolution, resolution), dtype=np.uint16)
    vertices_2d = (mesh.vertices[:, :2] - [min_x - offset_x, min_y - offset_y]) / pixel_size
    vertices_2d = np.clip(vertices_2d, 0, resolution - 1)


    for face in mesh.faces:
        pixels = np.round(vertices_2d[face]).astype(np.int32) # round the vertices to the nearest integer, may cause some loss of information
        cv2.fillPoly(projection, [pixels], 1)    
    

    # Initialize heightmap, set initial value to minimum height
    heightmap = np.full((resolution, resolution), fill_value=0, dtype=np.uint16)

    # Generate grid points (x, y)
    # x_lin = np.linspace(min_x, max_x, resolution)
    # y_lin = np.linspace(min_y, max_y, resolution)
    # xx, yy = np.meshgrid(x_lin, y_lin)
    # Use the same range as projection to build grid points
    x_lin = np.linspace(min_x - offset_x, min_x - offset_x + max_range, resolution)
    y_lin = np.linspace(min_y - offset_y, min_y - offset_y + max_range, resolution)
    xx, yy = np.meshgrid(x_lin, y_lin)


    # Build rays: origin (x, y, max_z + 1), direction (0, 0, -1) (downward projection)
    origins = np.stack([xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), max_z + 1)], axis=1) # generate grid points
    directions = np.array([[0, 0, -1]] * origins.shape[0])

    # Use Trimesh for ray casting
    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    hit_loc, _, _ = ray_intersector.intersects_location(origins, directions)

    # Parse intersection points of ray casting
    if hit_loc.shape[0] > 0:
        for i in range(hit_loc.shape[0]):
            x, y, z = hit_loc[i]
            #z = int(z*257)
            z_relative = int((z - min_z) * 257)
            # px = int((x - min_x) / pixel_size)
            # py = int((y - min_y) / pixel_size)
            px = int((x - (min_x - offset_x)) / pixel_size)
            py = int((y - (min_y - offset_y)) / pixel_size)
            if 0 <= px < resolution and 0 <= py < resolution:
                heightmap[py, px] = max(heightmap[py, px], z_relative)
    heightmap = heightmap[::-1]
    
    # Boundary in real coordinates
    bound_xy = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]])

    return heightmap, projection, mesh_faces, slop_area_ratio, slope_face_num, bound_xy, max_range, max_z,min_z

def slop_roof_detector(mesh):
    normals = mesh.face_normals

    significant_z_normals = np.abs(normals[:, 2]) > 0.3

    angle = np.arccos(np.abs(normals[:, 2]))
    inclined_faces = (angle > np.pi / 9) & (angle < np.pi / 2)

    slop_roof_faces = significant_z_normals & inclined_faces
    # Calculate slope area
    slopr_roof_indices = np.where(slop_roof_faces)[0]
    slope_face_num = len(slopr_roof_indices)
    slopr_roof_area = mesh.area_faces[slopr_roof_indices].sum()
    # Calculate the ratio of slope area to total area
    slopr_roof_area_ratio = slopr_roof_area / mesh.area_faces.sum()
    return slopr_roof_area_ratio, slope_face_num


