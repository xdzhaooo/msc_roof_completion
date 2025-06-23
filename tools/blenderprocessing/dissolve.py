import bpy
import os
import trimesh
import mathutils

# Set path and parameters
base_dir = r"D:\thesis\progress\3.15\dynamicresolution"
angle_limit = 0.0872665  # Approximately 5 degrees

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_ply_as_mesh(ply_path):
    # Read PLY file with trimesh
    mesh = trimesh.load(ply_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"Skipping non-triangular mesh: {ply_path}")
        return None

    vertices = [tuple(v) for v in mesh.vertices]
    faces = [tuple(f) for f in mesh.faces]

    # Create Blender mesh
    mesh_data = bpy.data.meshes.new(name=os.path.basename(ply_path))
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update()

    # Create Object and link to scene
    obj = bpy.data.objects.new(name="ImportedObject", object_data=mesh_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj

print("Start processing...")



def export_mesh_as_ply(obj, output_path):
    mesh = obj.data
    mesh.calc_loop_triangles()

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(mesh.polygons)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")

        # Write vertices
        for v in mesh.vertices:
            f.write(f"{v.co.x} {v.co.y} {v.co.z}\n")

        # Write faces (supports n-gon)
        for p in mesh.polygons:
            indices = " ".join(str(i) for i in p.vertices)
            f.write(f"{len(p.vertices)} {indices}\n")



for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    if os.path.isdir(subfolder_path):
        output_folder = os.path.join(subfolder_path, "Dissolved")
        os.makedirs(output_folder, exist_ok=True)

        mesh_path = os.path.join(subfolder_path, "mesh")
        if not os.path.exists(mesh_path):
            print(f"Skipping: mesh folder not found: {mesh_path}")
            continue

        for filename in os.listdir(mesh_path):
            if filename.lower().endswith(".ply"):
                ply_path = os.path.join(mesh_path, filename)
                print(f"Processing file: {ply_path}")

                clear_scene()
                obj = import_ply_as_mesh(ply_path)
                if obj is None:
                    print(f"Import failed: {ply_path}")
                    continue

                # Edit mode dissolve
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.dissolve_limited(angle_limit=angle_limit)
                bpy.ops.object.mode_set(mode='OBJECT')

                # Export
                output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_dissolved.ply")
                export_mesh_as_ply(obj, output_path)
                print(f"Export complete: {output_path}")


