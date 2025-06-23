#!/usr/bin/env python3
"""
Test script: Verify if the modified Network class can correctly load pretrained models
"""

import sys
import os
import yaml
import torch

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_network_loading():
    """Test network loading functionality"""
    
    try:
        # Read config file
        with open('mom.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Get network config
        network_config = config['model']['which_networks'][0]['args']
        
        print("=== Test Network class initialization ===")
        print(f"Config keys: {network_config.keys()}")
        
        try:
            # Import Network class
            from mom_models.network import Network
            
            # Create network instance
            network = Network(
                unetMom=network_config['unetMom'],
                beta_schedule=network_config['beta_schedule'],
                module_name=network_config['module_name']
            )
            
            print("✓ Network class initialized successfully")
            
            # Print initial parameter statistics
            print("\n=== Initial parameter statistics ===")
            try:
                network._print_trainable_parameters()
            except Exception as e:
                print(f"Error in parameter statistics: {e}")
                import traceback
                traceback.print_exc()
            
            # Test loading pretrained models
            print("\n=== Test loading pretrained models ===")
            
            roof_path = config['path']['pretrained_models']['roof_path']
            roofline_path = config['path']['pretrained_models']['roofline_path']
            
            print(f"Roof model path: {roof_path}")
            print(f"Roofline model path: {roofline_path}")
            
            # Check if files exist
            if not os.path.exists(roof_path):
                print(f"⚠ Roof model file does not exist: {roof_path}")
            else:
                print(f"✓ Roof model file exists")
                
            if not os.path.exists(roofline_path):
                print(f"⚠ Roofline model file does not exist: {roofline_path}")
            else:
                print(f"✓ Roofline model file exists")
                
            # Try loading models
            try:
                network.load_pretrained_models(roof_path, roofline_path)
                print("✓ Pretrained models loaded successfully")
                
                # Check if models loaded correctly
                if network.Roofdenoise_fn is not None:
                    print("✓ Roof model loaded successfully")
                else:
                    print("✗ Roof model loading failed")
                    
                if network.RoofLinedenoise_fn is not None:
                    print("✓ Roofline model loaded successfully")
                else:
                    print("✗ Roofline model loading failed")
                    
            except Exception as e:
                print(f"✗ Pretrained model loading failed: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n=== Test completed ===")
            
        except Exception as e:
            print(f"✗ Network class initialization failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Config file reading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_network_loading()