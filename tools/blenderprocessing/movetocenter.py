import sys
sys.path.append(r"C:\Users\zhaox\AppData\Roaming\Python\Python311\site-packages")
print("loaded")
import bpy
import os
import trimesh
import mathutils

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_ply_as_mesh(ply_path):
    mesh = trimesh.load(ply_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"Skipping non-triangular mesh: {ply_path}")
        return None
    
    vertices = [tuple(v) for v in mesh.vertices]
    faces = [tuple(f) for f in mesh.faces]
    
    mesh_data = bpy.data.meshes.new(name=os.path.basename(ply_path))
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update()
    
    obj = bpy.data.objects.new(name="ImportedObject", object_data=mesh_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj

def move_object_to_origin(obj):
    bpy.context.view_layer.update()
    
    # 获取包围盒
    bounds = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    
    # 计算中心点
    min_x = min([v[0] for v in bounds])
    max_x = max([v[0] for v in bounds])
    min_y = min([v[1] for v in bounds])
    max_y = max([v[1] for v in bounds])
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # 将物体移动到原点
    obj.location.x -= center_x
    obj.location.y -= center_y
    
    # 应用变换
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

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

        # 写入顶点
        for v in mesh.vertices:
            f.write(f"{v.co.x} {v.co.y} {v.co.z}\n")

        # 写入面（支持 n-gon）
        for p in mesh.polygons:
            indices = " ".join(str(i) for i in p.vertices)
            f.write(f"{len(p.vertices)} {indices}\n")

def process_folder(base_dir, angle_limit=0.087):  # Default angle is about 5 degrees
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # 创建输出文件夹
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
                    
                    # 编辑模式下 dissolve
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.mesh.dissolve_limited(angle_limit=angle_limit)
                    bpy.ops.object.mode_set(mode='OBJECT')
                    
                    # 移动到原点
                    move_object_to_origin(obj)
                    print(f"Moved {filename} to origin")
                    
                    # 导出到新位置
                    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_dissolved.ply")
                    export_mesh_as_ply(obj, output_path)
                    print(f"Export complete: {output_path}")


# 设置路径
base_dir = r"D:\thesis\progress\3.15\dynamicresolution"
process_folder(base_dir)
print("All models have been processed and moved to the origin!")