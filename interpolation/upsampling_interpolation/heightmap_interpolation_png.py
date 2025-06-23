#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heightmap Interpolation Tool - PNG Version
Supports four interpolation methods: IDW, Nearest, Spline, Perona-Malik Diffusion
Input and output are both PNG format
Added resize interpolation: interpolate 128x128 images to any target size
"""

import os
import glob
import numpy as np
import cv2
from scipy import interpolate
from scipy.spatial.distance import cdist
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class HeightmapInterpolator:
    """Heightmap Interpolator - PNG Version"""
    
    def __init__(self):
        self.supported_methods = ['idw', 'nearest', 'spline', 'perona_malik']
    
    def load_heightmap(self, filepath):
        """
        Load PNG heightmap, automatically handle 8-bit and 16-bit PNG
        """
        try:
            # Load PNG with PIL
            img = Image.open(filepath)
            
            # If 16-bit PNG, convert directly
            if img.mode in ['I;16', 'I']:
                heightmap = np.array(img, dtype=np.uint16)
            # If 8-bit PNG, expand to 16-bit
            elif img.mode in ['L', 'RGB', 'RGBA']:
                img_array = np.array(img)
                if len(img_array.shape) == 3:  # RGB/RGBA to grayscale
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                # Expand 8-bit to 16-bit (0-255 -> 0-65535)
                heightmap = (img_array.astype(np.float32) * 257).astype(np.uint16)
            else:
                # Other formats convert to uint16
                heightmap = np.array(img.convert('L'), dtype=np.uint16) * 257
            
            return heightmap
        except Exception as e:
            print(f"Error loading heightmap {filepath}: {e}")
            return None
    
    def load_heightmap_with_size_check(self, filepath, expected_size=(128, 128)):
        """
        Load heightmap and check size, resize if not as expected
        """
        heightmap = self.load_heightmap(filepath)
        if heightmap is None:
            return None
            
        # Resize if not expected size
        if heightmap.shape != expected_size:
            print(f"Warning: {filepath} size is {heightmap.shape}, resizing to {expected_size}")
            heightmap = cv2.resize(heightmap, expected_size, interpolation=cv2.INTER_NEAREST)
        
        return heightmap
    
    def get_image_size(self, filepath):
        """
        Get image size
        """
        try:
            img = Image.open(filepath)
            return img.size  # returns (width, height)
        except Exception as e:
            print(f"Error getting image size {filepath}: {e}")
            return None
    
    def load_footprint(self, filepath):
        """
        Load footprint PNG file (0 for non-region, 1 for valid region)
        """
        try:
            img = Image.open(filepath)
            
            # Convert to grayscale
            if img.mode != 'L':
                if img.mode in ['RGB', 'RGBA']:
                    img_array = np.array(img)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    footprint = img_array
                else:
                    footprint = np.array(img.convert('L'), dtype=np.uint8)
            else:
                footprint = np.array(img, dtype=np.uint8)
            
            # Resize if not 128x128
            if footprint.shape != (128, 128):
                print(f"Warning: {filepath} size is {footprint.shape}, resizing to 128x128")
                footprint = cv2.resize(footprint, (128, 128), interpolation=cv2.INTER_NEAREST)
            
            # Ensure only 0 and 1
            footprint = (footprint > 0).astype(np.uint8)
            return footprint
        except Exception as e:
            print(f"Error loading footprint {filepath}: {e}")
            return None

    def resize_interpolation(self, heightmap_128, target_size, method='bilinear'):
        """
        Interpolate 128x128 heightmap to target size
        
        Args:
            heightmap_128: 128x128 heightmap array
            target_size: target size (width, height)
            method: interpolation method ('nearest', 'bilinear', 'cubic', 'lanczos')
        
        Returns:
            Interpolated heightmap
        """
        if heightmap_128 is None:
            return None
        
        # OpenCV interpolation method mapping
        method_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        interpolation_method = method_map.get(method, cv2.INTER_LINEAR)
        
        # Resize
        resized = cv2.resize(heightmap_128, target_size, interpolation=interpolation_method)
        
        return resized.astype(np.uint16)

    def resize_with_advanced_interpolation(self, heightmap_128, target_size, method='idw', **kwargs):
        """
        Use advanced interpolation to resize 128x128 heightmap to target size
        
        Args:
            heightmap_128: 128x128 heightmap array
            target_size: target size (width, height)
            method: interpolation method ('idw', 'nearest', 'spline', 'perona_malik')
            **kwargs: extra arguments for interpolation method
        
        Returns:
            Interpolated heightmap
        """
        if heightmap_128 is None:
            return None
        
        # Get nonzero points as known points
        valid_mask = heightmap_128 > 0
        if not np.any(valid_mask):
            return np.zeros(target_size[::-1], dtype=np.uint16)  # note shape order
        
        y_coords, x_coords = np.where(valid_mask)
        values = heightmap_128[valid_mask]
        
        # Map coordinates to target size
        scale_x = target_size[0] / 128.0
        scale_y = target_size[1] / 128.0
        
        known_points = np.column_stack([
            x_coords * scale_x,
            y_coords * scale_y
        ])
        
        # Create target grid
        target_x = np.arange(target_size[0])
        target_y = np.arange(target_size[1])
        target_xx, target_yy = np.meshgrid(target_x, target_y)
        target_points = np.column_stack([target_xx.ravel(), target_yy.ravel()])
        
        # Interpolate by method
        if method == 'idw':
            power = kwargs.get('power', 2)
            max_distance = kwargs.get('max_distance', None)
            result = self._idw_interpolation_resize(known_points, values, target_points, target_size, power, max_distance)
        elif method == 'nearest':
            result = self._nearest_interpolation_resize(known_points, values, target_points, target_size)
        elif method == 'spline':
            result = self._spline_interpolation_resize(known_points, values, target_points, target_size)
        else:  # default to bilinear
            result = self.resize_interpolation(heightmap_128, target_size, 'bilinear')
        
        return result

    def _idw_interpolation_resize(self, known_points, values, target_points, target_size, power=2, max_distance=None):
        """IDW interpolation for resize"""
        result = np.zeros(len(target_points), dtype=np.float64)
        
        if max_distance is None:
            # Auto max distance based on target size
            max_distance = max(target_size) * 0.5
        
        for i, point in enumerate(target_points):
            distances = np.sqrt(np.sum((known_points - point) ** 2, axis=1))
            
            # Filter out far points
            valid_indices = distances <= max_distance
            if not np.any(valid_indices):
                continue
            
            distances = distances[valid_indices]
            point_values = values[valid_indices]
            
            # Handle zero distance
            if np.min(distances) < 1e-10:
                zero_dist_idx = np.argmin(distances)
                result[i] = point_values[zero_dist_idx]
            else:
                # IDW formula
                weights = 1.0 / (distances ** power)
                result[i] = np.sum(weights * point_values) / np.sum(weights)
        
        return result.reshape(target_size[1], target_size[0]).astype(np.uint16)

    def _nearest_interpolation_resize(self, known_points, values, target_points, target_size):
        """Nearest neighbor interpolation for resize"""
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(known_points)
        distances, indices = nbrs.kneighbors(target_points)
        
        result = values[indices.flatten()]
        return result.reshape(target_size[1], target_size[0]).astype(np.uint16)

    def _spline_interpolation_resize(self, known_points, values, target_points, target_size):
        """Spline interpolation for resize"""
        # For large images, limit spline interpolation to avoid memory issues
        max_pixels = 512 * 512  # max allowed 512x512
        if target_size[0] * target_size[1] > max_pixels:
            print(f"Target size {target_size} is too large for spline interpolation, using IDW instead")
            return self._idw_interpolation_resize(known_points, values, target_points, target_size, power=2)
        
        try:
            from scipy.interpolate import Rbf
            
            # Limit number of known points for performance
            max_known_points = 1000
            if len(known_points) > max_known_points:
                # Randomly sample known points
                indices = np.random.choice(len(known_points), max_known_points, replace=False)
                known_points = known_points[indices]
                values = values[indices]
            
            rbf = Rbf(known_points[:, 0], known_points[:, 1], values, 
                     function='thin_plate', smooth=0.1)  # add smoothing
            
            result = rbf(target_points[:, 0], target_points[:, 1])
            result = np.maximum(result, 0)  # ensure non-negative
            return result.reshape(target_size[1], target_size[0]).astype(np.uint16)
        
        except Exception as e:
            print(f"Spline interpolation failed: {e}, using IDW instead")
            return self._idw_interpolation_resize(known_points, values, target_points, target_size, power=2)
    
    def get_valid_points(self, heightmap, footprint):
        """
        Get valid points (footprint=1 and heightmap>0)
        """
        valid_mask = (footprint == 1) & (heightmap > 0)
        y_coords, x_coords = np.where(valid_mask)
        values = heightmap[valid_mask]
        return np.column_stack([x_coords, y_coords]), values
    
    def get_interpolation_points(self, footprint):
        """
        Get points to interpolate (all points where footprint=1)
        """
        y_coords, x_coords = np.where(footprint == 1)
        return np.column_stack([x_coords, y_coords])
    
    def normalize_to_8bit(self, image_16bit):
        """
        Normalize 16-bit image to 8-bit for visualization
        Normalize each image to its own range (min to max mapped to 0-255)
        """
        # Convert to float to avoid overflow
        img_float = image_16bit.astype(np.float32)
        
        # Find range of non-zero values (ignore background 0)
        non_zero_mask = img_float > 0
        if not np.any(non_zero_mask):
            # If all zeros, return all-zero 8-bit image
            return np.zeros_like(image_16bit, dtype=np.uint8)
        
        # Get min and max of non-zero values
        min_val = np.min(img_float[non_zero_mask])
        max_val = np.max(img_float[non_zero_mask])
        
        # Avoid division by zero
        if max_val == min_val:
            # If all non-zero values are the same, set to mid-gray
            result = np.zeros_like(image_16bit, dtype=np.uint8)
            result[non_zero_mask] = 128
            return result
        
        # Create output image
        result = np.zeros_like(image_16bit, dtype=np.uint8)
        
        # Normalize non-zero region
        normalized = (img_float[non_zero_mask] - min_val) / (max_val - min_val)
        result[non_zero_mask] = (normalized * 255).astype(np.uint8)
        
        return result
    
    def idw_interpolation(self, heightmap, footprint, power=2, max_distance=50):
        """
        Inverse Distance Weighting (IDW) interpolation
        """
        known_points, known_values = self.get_valid_points(heightmap, footprint)
        if len(known_points) == 0:
            return heightmap.copy()
        
        interpolation_points = self.get_interpolation_points(footprint)
        result = heightmap.copy().astype(np.float64)
        
        for point in interpolation_points:
            # Compute distance to all known points
            distances = np.sqrt(np.sum((known_points - point) ** 2, axis=1))
            
            # Skip if point already has value
            if result[point[1], point[0]] > 0:
                continue
            
            # Filter out points that are too far
            valid_indices = distances <= max_distance
            if not np.any(valid_indices):
                continue
            
            distances = distances[valid_indices]
            values = known_values[valid_indices]
            
            # Handle zero distance
            if np.min(distances) < 1e-10:
                zero_dist_idx = np.argmin(distances)
                result[point[1], point[0]] = values[zero_dist_idx]
            else:
                # IDW formula
                weights = 1.0 / (distances ** power)
                result[point[1], point[0]] = np.sum(weights * values) / np.sum(weights)
        
        return result.astype(np.uint16)
    
    def nearest_interpolation(self, heightmap, footprint):
        """
        Nearest neighbor interpolation
        """
        known_points, known_values = self.get_valid_points(heightmap, footprint)
        if len(known_points) == 0:
            return heightmap.copy()
        
        interpolation_points = self.get_interpolation_points(footprint)
        result = heightmap.copy().astype(np.float64)
        
        # Use sklearn NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(known_points)
        
        for point in interpolation_points:
            # Skip if point already has value
            if result[point[1], point[0]] > 0:
                continue
            
            distances, indices = nbrs.kneighbors([point])
            nearest_value = known_values[indices[0][0]]
            result[point[1], point[0]] = nearest_value
        
        return result.astype(np.uint16)
    
    def spline_interpolation(self, heightmap, footprint, method='cubic'):
        """
        Spline interpolation
        """
        known_points, known_values = self.get_valid_points(heightmap, footprint)
        if len(known_points) < 4:  # Spline interpolation needs enough points
            print("Warning: Not enough points for spline interpolation, using IDW instead")
            return self.idw_interpolation(heightmap, footprint)
        
        interpolation_points = self.get_interpolation_points(footprint)
        result = heightmap.copy().astype(np.float64)
        
        try:
            # Use Rbf interpolation
            from scipy.interpolate import Rbf
            rbf = Rbf(known_points[:, 0], known_points[:, 1], known_values, 
                     function='thin_plate', smooth=0)
            
            for point in interpolation_points:
                # Skip if point already has value
                if result[point[1], point[0]] > 0:
                    continue
                
                interpolated_value = rbf(point[0], point[1])
                result[point[1], point[0]] = max(0, interpolated_value)
        
        except Exception as e:
            print(f"Spline interpolation failed: {e}, using IDW instead")
            return self.idw_interpolation(heightmap, footprint)
        
        return result.astype(np.uint16)
    
    def perona_malik_diffusion(self, heightmap, footprint, iterations=100, delta_t=0.1, kappa=15):
        """
        Perona-Malik diffusion interpolation
        """
        result = heightmap.copy().astype(np.float64)
        
        # Create mask, only diffuse in footprint region
        valid_mask = (footprint == 1)
        
        for i in range(iterations):
            # Compute gradients
            grad_north = np.roll(result, -1, axis=0) - result
            grad_south = np.roll(result, 1, axis=0) - result
            grad_east = np.roll(result, -1, axis=1) - result
            grad_west = np.roll(result, 1, axis=1) - result
            
            # Compute diffusion coefficients (Perona-Malik)
            c_north = np.exp(-(grad_north / kappa) ** 2)
            c_south = np.exp(-(grad_south / kappa) ** 2)
            c_east = np.exp(-(grad_east / kappa) ** 2)
            c_west = np.exp(-(grad_west / kappa) ** 2)
            
            # Update
            diff = delta_t * (c_north * grad_north + c_south * grad_south + 
                             c_east * grad_east + c_west * grad_west)
            
            # Only update in valid region, keep known points unchanged
            update_mask = valid_mask & (heightmap == 0)
            result[update_mask] += diff[update_mask]
            
            # Ensure no negative values
            result = np.maximum(result, 0)
        
        return result.astype(np.uint16)
    
    def interpolate_single(self, heightmap_path, footprint_path, method='idw', save_8bit_vis=False, output_dir=None, **kwargs):
        """
        Interpolate a single file
        
        Args:
            heightmap_path: heightmap file path
            footprint_path: footprint file path
            method: interpolation method
            save_8bit_vis: whether to save 8-bit visualization version
            output_dir: output directory (if saving 8-bit version)
            **kwargs: extra arguments for interpolation method
        """
        # Load data
        heightmap = self.load_heightmap(heightmap_path)
        footprint = self.load_footprint(footprint_path)
        
        if heightmap is None or footprint is None:
            return None
        
        # Interpolate according to method
        if method == 'idw':
            result = self.idw_interpolation(heightmap, footprint, **kwargs)
        elif method == 'nearest':
            result = self.nearest_interpolation(heightmap, footprint)
        elif method == 'spline':
            result = self.spline_interpolation(heightmap, footprint, **kwargs)
        elif method == 'perona_malik':
            result = self.perona_malik_diffusion(heightmap, footprint, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Save 8-bit visualization if needed
        if save_8bit_vis and output_dir is not None and result is not None:
            basename = os.path.splitext(os.path.basename(heightmap_path))[0]
            
            # Create 8-bit visualization directory
            output8bit_dir = os.path.join(output_dir, "8bit_visualization")
            os.makedirs(output8bit_dir, exist_ok=True)
            
            # Generate 8-bit visualization
            result_8bit = self.normalize_to_8bit(result)
            output8bit_path = os.path.join(output8bit_dir, f"{basename}_{method}_vis.png")
            result_8bit_img = Image.fromarray(result_8bit, mode='L')
            result_8bit_img.save(output8bit_path)
            print(f"8-bit visualization saved: {output8bit_path}")
        
        return result
    
    def batch_interpolate(self, heightmap_dir, footprint_dir, output_dir, method='idw', **kwargs):
        """
        Batch interpolation processing
        
        Args:
            heightmap_dir: heightmap file directory
            footprint_dir: footprint file directory
            output_dir: output directory
            method: interpolation method ('idw', 'nearest', 'spline', 'perona_malik')
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all heightmap files
        heightmap_files = glob.glob(os.path.join(heightmap_dir, "*"))
        heightmap_files = [f for f in heightmap_files if os.path.isfile(f)]
        
        processed_count = 0
        failed_count = 0
        
        print(f"Starting batch interpolation using {method} method...")
        print(f"Found {len(heightmap_files)} heightmap files")
        
        for heightmap_path in heightmap_files:
            # Get filename without extension
            basename = os.path.splitext(os.path.basename(heightmap_path))[0]
            
            # Find corresponding footprint file (PNG format)
            footprint_patterns = [
                os.path.join(footprint_dir, f"{basename}.png"),
                os.path.join(footprint_dir, f"{basename}.*")
            ]
            
            footprint_path = None
            for pattern in footprint_patterns:
                matches = glob.glob(pattern)
                if matches:
                    footprint_path = matches[0]
                    break
            
            if footprint_path is None:
                print(f"Warning: No footprint file found for {basename}")
                failed_count += 1
                continue
            
            # Perform interpolation
            try:
                result = self.interpolate_single(heightmap_path, footprint_path, method, **kwargs)
                if result is not None:
                    # Save 16-bit PNG result
                    output_path = os.path.join(output_dir, f"{basename}_{method}.png")
                    result_img = Image.fromarray(result, mode='I;16')
                    result_img.save(output_path)
                    
                    # Create 8-bit visualization version
                    output8bit_dir = output_dir.replace("output", "output8bit")
                    os.makedirs(output8bit_dir, exist_ok=True)
                    
                    # Normalize each image to 8-bit
                    result_8bit = self.normalize_to_8bit(result)
                    output8bit_path = os.path.join(output8bit_dir, f"{basename}_{method}_vis.png")
                    result_8bit_img = Image.fromarray(result_8bit, mode='L')
                    result_8bit_img.save(output8bit_path)
                    
                    processed_count += 1
                    print(f"Processed: {basename} -> {output_path}")
                    print(f"          8-bit vis: {output8bit_path}")
                else:
                    failed_count += 1
                    print(f"Failed to process: {basename}")
            
            except Exception as e:
                failed_count += 1
                print(f"Error processing {basename}: {e}")
        
        print(f"\nBatch processing completed!")
        print(f"Successfully processed: {processed_count} files")
        print(f"Failed: {failed_count} files")


def main():
    """
    Main function - usage example
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Heightmap Interpolation Tool - PNG Version')
    parser.add_argument('--heightmap_dir', required=True, help='Heightmap PNG files directory')
    parser.add_argument('--footprint_dir', required=True, help='Footprint PNG files directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--method', choices=['idw', 'nearest', 'spline', 'perona_malik'], 
                       default='idw', help='Interpolation method')
    parser.add_argument('--power', type=float, default=2.0, help='Power for IDW (default: 2.0)')
    parser.add_argument('--max_distance', type=float, default=50.0, help='Max distance for IDW (default: 50.0)')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations for Perona-Malik (default: 100)')
    parser.add_argument('--kappa', type=float, default=15.0, help='Kappa for Perona-Malik (default: 15.0)')
    
    args = parser.parse_args()
    
    # Create interpolator
    interpolator = HeightmapInterpolator()
    
    # Prepare parameters
    kwargs = {}
    if args.method == 'idw':
        kwargs = {'power': args.power, 'max_distance': args.max_distance}
    elif args.method == 'perona_malik':
        kwargs = {'iterations': args.iterations, 'kappa': args.kappa}
    
    # Run batch interpolation
    interpolator.batch_interpolate(
        args.heightmap_dir,
        args.footprint_dir,
        args.output_dir,
        args.method,
        **kwargs
    )


if __name__ == "__main__":
    main()