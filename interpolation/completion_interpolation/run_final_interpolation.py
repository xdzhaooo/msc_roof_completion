#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from heightmap_interpolation_png import HeightmapInterpolator
import os
import time
import glob
from PIL import Image

class FixedHeightmapInterpolator(HeightmapInterpolator):
    """Fixed version of the interpolator, handles matching filenames with _footprint suffix"""
    
    def batch_interpolate(self, heightmap_dir, footprint_dir, output_dir, method='idw', **kwargs):
        """
        Batch interpolation processing
        Fixed version: correctly handles matching filenames with _footprint suffix
        """
        # Create output directories
        method_output_dir = os.path.join(output_dir, f"output_{method}")
        method_output8bit_dir = os.path.join(output_dir, f"output8bit_{method}")
        os.makedirs(method_output_dir, exist_ok=True)
        os.makedirs(method_output8bit_dir, exist_ok=True)
        
        # Get all heightmap files
        heightmap_files = glob.glob(os.path.join(heightmap_dir, "*.png"))
        
        if not heightmap_files:
            print(f"No PNG files found in {heightmap_dir}")
            return 0, 0
        
        print(f"Found {len(heightmap_files)} heightmap files")
        
        success_count = 0
        fail_count = 0
        
        for heightmap_path in heightmap_files:
            try:
                # Get base filename (without extension)
                base_name = os.path.splitext(os.path.basename(heightmap_path))[0]
                
                # Construct corresponding footprint file path (add _footprint suffix)
                footprint_name = f"{base_name}_footprint.png"
                footprint_path = os.path.join(footprint_dir, footprint_name)
                
                # Check if footprint file exists
                if not os.path.exists(footprint_path):
                    print(f"Warning: Corresponding footprint file not found: {footprint_name}")
                    fail_count += 1
                    continue
                
                # Construct output file paths
                output_filename = f"{base_name}_{method}_interpolated.png"
                output_path = os.path.join(method_output_dir, output_filename)
                output8bit_path = os.path.join(method_output8bit_dir, output_filename)
                
                # Perform interpolation - use correct method call
                result = self.interpolate_single(
                    heightmap_path=heightmap_path,
                    footprint_path=footprint_path,
                    method=method,
                    **kwargs
                )
                
                if result is not None:
                    # Save 16-bit PNG result
                    result_img = Image.fromarray(result, mode='I;16')
                    result_img.save(output_path)
                    
                    # Generate 8-bit visualization
                    result_8bit = self.normalize_to_8bit(result)
                    result_8bit_img = Image.fromarray(result_8bit, mode='L')
                    result_8bit_img.save(output8bit_path)
                    
                    success_count += 1
                    if success_count % 10 == 0:
                        print(f"Processed {success_count}/{len(heightmap_files)} files...")
                else:
                    fail_count += 1
                    
            except Exception as e:
                print(f"Error processing file {os.path.basename(heightmap_path)}: {e}")
                fail_count += 1
        
        print(f"\nBatch processing completed!")
        print(f"Successfully processed: {success_count} files")
        print(f"Failed: {fail_count} files")
        
        return success_count, fail_count

def main():
    # User-specified paths
    heightmap_dir = r"benchmark\w_footprint\s95_i30\roof_img"
    footprint_dir = r"benchmark\w_footprint\s95_i30\roof_footprint" 
    output_dir = r"benchmark\w_footprint\s95_i30"
    
    print("Heightmap Interpolation Tool - Final Fixed Version")
    print("=" * 60)
    print(f"Heightmap folder: {heightmap_dir}")
    print(f"Footprint folder: {footprint_dir}")
    print(f"Output folder: {output_dir}")
    print("=" * 60)
    
    # Check if folders exist
    if not os.path.exists(heightmap_dir):
        print(f"Error: Heightmap folder does not exist: {heightmap_dir}")
        return
    
    if not os.path.exists(footprint_dir):
        print(f"Error: Footprint folder does not exist: {footprint_dir}")
        return
    
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize fixed version interpolator
    interpolator = FixedHeightmapInterpolator()
    
    # Run all interpolation methods
    methods = ['idw', 'nearest', 'spline', 'perona_malik']
    
    total_stats = {}
    overall_start_time = time.time()
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()} interpolation...")
        print('='*60)
        
        start_time = time.time()
        
        try:
            success_count, fail_count = interpolator.batch_interpolate(
                heightmap_dir=heightmap_dir,
                footprint_dir=footprint_dir, 
                output_dir=output_dir,
                method=method
            )
            
            elapsed_time = time.time() - start_time
            total_stats[method] = {
                'success': success_count,
                'fail': fail_count,
                'time': elapsed_time
            }
            
            print(f"\n{method.upper()} interpolation completed!")
            print(f"Success: {success_count} files")
            print(f"Fail: {fail_count} files") 
            print(f"Time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"\nError occurred during {method.upper()} interpolation: {e}")
            total_stats[method] = {
                'success': 0,
                'fail': 0,
                'time': 0,
                'error': str(e)
            }
    
    # Show summary
    overall_elapsed_time = time.time() - overall_start_time
    print("\n" + "="*60)
    print("All interpolation methods completed!")
    print("="*60)
    print("Summary of results:")
    
    total_success = 0
    total_fail = 0
    
    for method, stats in total_stats.items():
        if 'error' in stats:
            print(f"  {method.upper()}: Error occurred - {stats['error']}")
        else:
            print(f"  {method.upper()}: Success {stats['success']} files, Fail {stats['fail']} files, Time {stats['time']:.2f} seconds")
            total_success += stats['success']
            total_fail += stats['fail']
    
    print(f"\nTotal: Success {total_success} files, Fail {total_fail} files")
    print(f"Total time: {overall_elapsed_time:.2f} seconds")
    
    # Show output folder structure
    print(f"\nOutput files are located in: {output_dir}")
    try:
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('output')]
        if subdirs:
            print("Generated subfolders:")
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(output_dir, subdir)
                file_count = len([f for f in os.listdir(subdir_path) if f.endswith('.png')])
                print(f"  - {subdir}: {file_count} files")
    except Exception as e:
        print(f"Could not list output folder contents: {e}")

if __name__ == "__main__":
    main()