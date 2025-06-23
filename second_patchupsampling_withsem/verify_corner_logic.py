#!/usr/bin/env python3
"""
Verify the logic that patch_pos returns the top-left corner coordinates
"""

import sys
sys.path.append('.')

from data.datasetPatch import RoofPatchDataset
import numpy as np
from PIL import Image
import os

def create_test_data():
    """Create test data"""
    test_dir = "test_corner"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create 256x256 test image
    test_img = np.random.rand(256, 256) * 255
    test_img = test_img.astype(np.uint8)
    
    Image.fromarray(test_img).save(f"{test_dir}/test_gt.png")
    Image.fromarray(test_img).save(f"{test_dir}/test_corrupted.png")
    Image.fromarray(test_img).save(f"{test_dir}/test_footprint.png")
    Image.fromarray(test_img).save(f"{test_dir}/test_first_output.png")
    
    current_dir = os.path.abspath(".")
    
    with open(f"{test_dir}/gt.flist", "w") as f:
        f.write(f"{current_dir}/{test_dir}/test_gt.png")
    with open(f"{test_dir}/corrupted.flist", "w") as f:
        f.write(f"{current_dir}/{test_dir}/test_corrupted.png")
    with open(f"{test_dir}/footprint.flist", "w") as f:
        f.write(f"{current_dir}/{test_dir}/test_footprint.png")
    with open(f"{test_dir}/first_output.flist", "w") as f:
        f.write(f"{current_dir}/{test_dir}/test_first_output.png")
    
    return test_dir

def verify_corner_positions():
    """Verify top-left corner coordinate logic"""
    print("="*80)
    print("Verify that patch_pos returns top-left corner coordinates (was center coordinates before modification)")
    print("="*80)
    
    test_dir = create_test_data()
    
    try:
        dataset = RoofPatchDataset(
            corrupted_root=f"{test_dir}/corrupted.flist",
            data_root=f"{test_dir}/gt.flist",
            footprint_root=f"{test_dir}/footprint.flist",
            first_output_root=f"{test_dir}/first_output.flist",
            patch_size=[128, 128],
            image_size=[256, 256],
            mask_config={"down_res_pct": [0], "local_remove": [[0, 0, 0]]},
            noise_config={},
            data_aug={"repeat": 1, "rotate": 360}
        )
        
        print(f"Dataset settings:")
        print(f"  Original image size: 256x256")
        print(f"  patch_size: 128x128")
        print(f"  Large patch size returned: 256x256 (contains 2x2 small 128x128 patches)")
        print(f"  Stride: 128")
        print(f"  Padding: 64 pixels")
        print(f"  Image size after padding: 384x384")
        
        print(f"\nTheoretical analysis:")
        print(f"  For 256x256 image, stride 128, large patch 256x256")
        print(f"  Number of large patches that can be generated: ((256-256)//128+1)² = 1² × 1² = 2×2 = 4")
        print(f"  Actual number of patches in dataset: {len(dataset)}")
        
        # Test the first patch to verify logic
        sample = dataset[0]
        patch_pos = sample['patch_pos'].numpy()
        
        print(f"\nDetailed analysis of patch 0:")
        print(f"  patch_pos.shape: {patch_pos.shape}")
        print(f"  patch_pos content (normalized coordinates):")
        
        for i in range(2):
            for j in range(2):
                norm_y, norm_x = patch_pos[i, j]
                print(f"    small patch[{i},{j}]: ({norm_y:.3f}, {norm_x:.3f})")
        
        print(f"\nVerification: If it is the top-left corner coordinate (not center):")
        print(f"  The starting position of the 0th large patch in the padded image: (0, 0)")  # patch_pos = (0,0), so 0*step = (0,0)
        print(f"  The 2x2 small patches' top-left positions should be:")
        print(f"    small patch[0,0]: padded image(0, 0) -> normalized({0/384:.3f}, {0/384:.3f})")
        print(f"    small patch[0,1]: padded image(0, 128) -> normalized({0/384:.3f}, {128/384:.3f})")
        print(f"    small patch[1,0]: padded image(128, 0) -> normalized({128/384:.3f}, {0/384:.3f})")
        print(f"    small patch[1,1]: padded image(128, 128) -> normalized({128/384:.3f}, {128/384:.3f})")
        
        print(f"\nCompare with actual results:")
        expected = [(0/384, 0/384), (0/384, 128/384), (128/384, 0/384), (128/384, 128/384)]
        idx = 0
        all_match = True
        for i in range(2):
            for j in range(2):
                actual = (patch_pos[i, j, 0], patch_pos[i, j, 1])
                exp = expected[idx]
                match = abs(actual[0] - exp[0]) < 1e-6 and abs(actual[1] - exp[1]) < 1e-6
                status = "✅ Match" if match else "❌ Mismatch"
                print(f"    small patch[{i},{j}]: actual({actual[0]:.3f}, {actual[1]:.3f}) vs expected({exp[0]:.3f}, {exp[1]:.3f}) - {status}")
                if not match:
                    all_match = False
                idx += 1
        
        print(f"\n{'='*80}")
        if all_match:
            print("✅ Verification successful: patch_pos now returns top-left corner coordinates!")
            print("✅ Previously returned center coordinates, now correctly returns top-left corner coordinates")
        else:
            print("❌ Verification failed: coordinates do not match")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    verify_corner_positions()