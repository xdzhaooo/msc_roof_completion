from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.file_utils import scan_files, write_to_png
from utils.mesh_footprintProcessor import project_mesh_to_xy
from utils.lasProcessor import convert_las_to_roof, cal_point_mean_dis
from utils.incompletenessCalculator import cal_sparsity, cal_chamfer_hausdorff_distance
import numpy as np
import os
import pandas as pd
import trimesh
from tqdm import tqdm
import sys


def process_file(k, v, path1, path2):
    try:
        laspath = v["ext1"]
        plypath = v["ext2"]

        # Load and process the mesh
        mesh = trimesh.load_mesh(plypath)
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        min_x, min_y = bounds[0][:2]
        max_x, max_y = bounds[1][:2]
        x_range = max_x - min_x
        y_range = max_y - min_y
        max_range = max(x_range, y_range)
        resolution = (max_range / 0.3).astype(int)
        resolution = (max_range / 0.4).astype(int)
        range_bbx = [max_range,min_x,max_x, min_y, max_y]

        avg_distance = cal_point_mean_dis(laspath)
        #print(avg_distance,"avg_distance")
        pixelsize = avg_distance*1.5
        resolution = (max_range / pixelsize).astype(int)
        range_bbx = [max_range,min_x,max_x, min_y, max_y]

        las_roof_np = convert_las_to_roof(laspath,range_bbx=range_bbx, resolution=resolution)
        las_roof_np = las_roof_np[::-1]
        las_roof_np[las_roof_np != 0] = 255 # 0 or 255
        #write_to_png(las_roof_np, os.path.join(path2, f"{k}_roof.png"), mode="L")

        
        # Project the mesh to the XY plane,setting range_bbx to is to ensure the height map has the same
        # range as the footprint file
        _, footprint_np, mesh_faces, slope_ratio, slope_face_num,_, max_range,max_z = project_mesh_to_xy(plypath, resolution=resolution)
        footprint_np = footprint_np[::-1]
        # footprint_area = np.sum(footprint_np)*0.3*0.3
        # footprint_area = np.sum(footprint_np)*0.4*0.4
        footprint_area = np.sum(footprint_np)*pixelsize*pixelsize
        footprint_np[footprint_np != 0] = 255 # 0 or 255
        #write_to_png(footprint_np, os.path.join(path2, f"{k}.png"), mode="L")


        sparsity_rate = cal_sparsity(footprint_np, las_roof_np)
        #chamfer_distance, hausdorff_distance = cal_chamfer_hausdorff_distance(footprint_np, las_roof_np)

        # path2=r"D:\thesis\progress\2.27\test2"
        # footprint_np = np.where(footprint_np>1,255,0).astype(np.uint8)
        # las_roof_np = np.where(las_roof_np>1,255,0).astype(np.uint8)
        # write_to_png(footprint_np, os.path.join(path2, f"{sparsity_rate}{k}.png"), mode="L")
        # write_to_png(las_roof_np, os.path.join(path2, f"{sparsity_rate}{k}_roof.png"),mode="L")

        normalized_chamfer_distance, normalized_hausdorff_distance, chamfer_distance, hausdorff_distance = cal_chamfer_hausdorff_distance(footprint_np, las_roof_np)


        return {"filename": k, "mesh_faces": mesh_faces,"footprint_area": footprint_area, "slope_ratio": slope_ratio,
                "slope_face_number": slope_face_num, "sparsity_rate": sparsity_rate, "chamfer_distance": chamfer_distance,
                "hausdorff_distance": hausdorff_distance, "max_range": max_range, "max_z": max_z,"normalized_chamfer_distance":normalized_chamfer_distance,"normalized_hausdorff_distance":normalized_hausdorff_distance}
    except Exception as e:
        print(f"Error processing file {k}: {e}")
        return None


# Main execution
if __name__ == "__main__":
    # path1 = r"D:\BuildingPccdata\Rotterdam_processed\output_all\matched_ahn4_pcl"
    # path2 = r"D:\BuildingPccdata\Rotterdam_processed\output_all\matched_gt_model"
    # path1 = r"D:\thesis\progress\12.15"
    # path2 = r"D:\thesis\progress\12.15"
    path_dic={
              "rotterdam_ahn3":[r"D:\BuildingPccdata\Rotterdam_processed\output_all\matched_ahn3_pcl",r"D:\BuildingPccdata\Rotterdam_processed\output_all\repaired"],
              "rotterdam_ahn4":[r"D:\BuildingPccdata\Rotterdam_processed\output_all\matched_ahn4_pcl",r"D:\BuildingPccdata\Rotterdam_processed\output_all\repaired"],
              "rotterdam_added_ahn3":[r"D:\BuildingPccdata\Rotterdam_processed\output_all\added_ahn3_las",r"D:\BuildingPccdata\Rotterdam_processed\output_all\repaired"],
              "rotterdam_added_ahn4":[r"D:\BuildingPccdata\Rotterdam_processed\output_all\added_ahn4_las",r"D:\BuildingPccdata\Rotterdam_processed\output_all\repaired"],
            
              "denhaag_ahn3":[r"D:\BuildingPccdata\DenHaag_processed\output_all\matched_ahn3_pcl",r"D:\BuildingPccdata\DenHaag_processed\output_all\repaired"],
              "denhaag_ahn4":[r"D:\BuildingPccdata\DenHaag_processed\output_all\matched_ahn4_pcl",r"D:\BuildingPccdata\DenHaag_processed\output_all\repaired"],
              "denhaag_added_ahn3":[r"D:\BuildingPccdata\DenHaag_processed\output_all\added_ahn3_las",r"D:\BuildingPccdata\DenHaag_processed\output_all\repaired"],
              "denhaag_added_ahn4":[r"D:\BuildingPccdata\DenHaag_processed\output_all\added_ahn4_las",r"D:\BuildingPccdata\DenHaag_processed\output_all\repaired"]
              }
    

    total_num = 1000 # Limit the number of files to process
    for city_name,paths in path_dic.items():



        path1=paths[0]
        path2=paths[1]
        #files = scan_files(path1, path2)
        print(f"Processing {city_name}")

        files = scan_files(path1, path2)  # path1: las, path2: ply
        print(f"Found {len(files)} files to process")

        rows = []
        total_files = len(files)  # Total number of files
        completed_count = 0  # Number of completed tasks
        file1 = files.popitem()
        # process_file(file1[0], file1[1], path1, path2)
        # print(f"Processing {file1[0]}")
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(process_file, k, v, path1, path2): k for k, v in files.items()}
            with tqdm(total=total_files, desc="Processing files") as pbar:  # Use tqdm to display a progress bar
                for future in as_completed(futures):

                    try:
                        result = future.result()
                        if result:
                            rows.append(result)
                            completed_count += 1  # Increment the completed count
                            pbar.update(1)  # Update the progress bar
                    except Exception as e:
                        print(f"Error in task {futures[future]}: {e}")
                        
                        pbar.update(1)  # Also update the progress bar if the task fails

                    # Print the number of completed files in real time
                    # if completed_count % 100 == 0:
                    #     print(f"Completed {completed_count}/{total_files} files")

        # Save results
        df = pd.DataFrame(rows)
        output_file = r"D:\thesis\progress\2.27"
        os.makedirs(output_file, exist_ok=True)
        output_path = os.path.join(output_file, f"{city_name}.parquet")
        df.to_parquet(output_path)
