#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from heightmap_interpolation_png import HeightmapInterpolator
import os
import time
import glob
from PIL import Image

class ResizeHeightmapInterpolator(HeightmapInterpolator):
    """Class for handling resize interpolation"""
    
    def batch_resize_interpolate(self, gt128_dir, target_gt_dir, output_dir, method='bilinear', **kwargs):
        """
        Batch resize interpolation processing
        Interpolates 128x128 images in gt128_dir to the real size of corresponding files in target_gt_dir
        
        Args:
            gt128_dir: folder containing 128x128 images
            target_gt_dir: folder containing real size images (for target size reference)
            output_dir: output folder
            method: interpolation method ('bilinear', 'cubic', 'nearest', 'lanczos', 'idw', 'spline')
        """
        # Create output directories
        method_output_dir = os.path.join(output_dir, f"resize_{method}")
        method_output8bit_dir = os.path.join(output_dir, f"resize8bit_{method}")
        os.makedirs(method_output_dir, exist_ok=True)
        os.makedirs(method_output8bit_dir, exist_ok=True)
        
        # Get all gt128 files
        gt128_files = glob.glob(os.path.join(gt128_dir, "*.png"))
        
        if not gt128_files:
            print(f"No PNG files found in {gt128_dir}")
            return 0, 0
        
        print(f"Found {len(gt128_files)} 128x128 files")
        
        success_count = 0
        fail_count = 0
        
        for gt128_path in gt128_files:
            try:
                # Get base filename (without extension)
                base_name = os.path.splitext(os.path.basename(gt128_path))[0]
                
                # Construct corresponding real size file path
                target_gt_path = os.path.join(target_gt_dir, f"{base_name}.png")
                
                # Check if target file exists
                if not os.path.exists(target_gt_path):
                    print(f"Warning: Corresponding target file not found: {base_name}.png")
                    fail_count += 1
                    continue
                
                # Load 128x128 heightmap
                heightmap_128 = self.load_heightmap_with_size_check(gt128_path, (128, 128))
                if heightmap_128 is None:
                    print(f"Warning: Could not load 128x128 file: {base_name}")
                    fail_count += 1
                    continue
                
                # Get target size
                target_size = self.get_image_size(target_gt_path)
                if target_size is None:
                    print(f"Warning: Could not get target file size: {base_name}")
                    fail_count += 1
                    continue
                
                print(f"Processing {base_name}: 128x128 -> {target_size}")
                
                # Perform resize interpolation
                if method in ['bilinear', 'cubic', 'nearest', 'lanczos']:
                    # Use OpenCV basic interpolation methods
                    result = self.resize_interpolation(heightmap_128, target_size, method)
                elif method in ['idw', 'spline']:
                    # Use advanced interpolation methods
                    result = self.resize_with_advanced_interpolation(heightmap_128, target_size, method, **kwargs)
                else:
                    print(f"Unsupported interpolation method: {method}")
                    fail_count += 1
                    continue
                
                if result is not None:
                    # Construct output file paths
                    output_filename = f"{base_name}_resize_{method}.png"
                    output_path = os.path.join(method_output_dir, output_filename)
                    output8bit_path = os.path.join(method_output8bit_dir, output_filename)
                    
                    # Save 16-bit PNG result
                    result_img = Image.fromarray(result, mode='I;16')
                    result_img.save(output_path)
                    
                    # Generate 8-bit visualization
                    result_8bit = self.normalize_to_8bit(result)
                    result_8bit_img = Image.fromarray(result_8bit, mode='L')
                    result_8bit_img.save(output8bit_path)
                    
                    success_count += 1
                    if success_count % 10 == 0:
                        print(f"Processed {success_count}/{len(gt128_files)} files...")
                else:
                    fail_count += 1
                    
            except Exception as e:
                print(f"Error processing file {os.path.basename(gt128_path)}: {e}")
                fail_count += 1
        
        print(f"\nBatch resize interpolation completed!")
        print(f"Successfully processed: {success_count} files")
        print(f"Failed: {fail_count} files")
        
        return success_count, fail_count

def main():
    # User-specified paths
    gt128_dir = r"gt128"  # 128x128 image folder
    target_gt_dir = r"s80_i80\roof_gt"  # real size image folder
    output_dir = r"resize_output"  # output folder
    
    print("Heightmap Resize Interpolation Tool")
    print("=" * 60)
    print(f"128x128 folder: {gt128_dir}")
    print(f"Target size reference folder: {target_gt_dir}")
    print(f"Output folder: {output_dir}")
    print("=" * 60)
    
    # Check if folders exist
    if not os.path.exists(gt128_dir):
        print(f"Error: 128x128 folder does not exist: {gt128_dir}")
        return
    
    if not os.path.exists(target_gt_dir):
        print(f"Error: Target size reference folder does not exist: {target_gt_dir}")
        return
    
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize interpolator
    interpolator = ResizeHeightmapInterpolator()
    
    # Run all interpolation methods
    methods = [
        ('bilinear', {}),
        ('cubic', {}),
        ('nearest', {}),
        ('lanczos', {}),
        ('idw', {'power': 2}),
        ('spline', {})
    ]
    
    total_stats = {}
    overall_start_time = time.time()
    
    for method, method_kwargs in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()} resize interpolation...")
        print('='*60)
        
        start_time = time.time()
        
        try:
            success_count, fail_count = interpolator.batch_resize_interpolate(
                gt128_dir=gt128_dir,
                target_gt_dir=target_gt_dir,
                output_dir=output_dir,
                method=method,
                **method_kwargs
            )
            
            elapsed_time = time.time() - start_time
            total_stats[method] = {
                'success': success_count,
                'fail': fail_count,
                'time': elapsed_time
            }
            
            print(f"\n{method.upper()} resize interpolation completed!")
            print(f"Success: {success_count} files")
            print(f"Fail: {fail_count} files") 
            print(f"Time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"\nError occurred during {method.upper()} resize interpolation: {e}")
            total_stats[method] = {
                'success': 0,
                'fail': 0,
                'time': 0,
                'error': str(e)
            }
    
    # Show summary
    overall_elapsed_time = time.time() - overall_start_time
    print("\n" + "="*60)
    print("All resize interpolation methods completed!")
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
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('resize')]
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