# Structure Guided Roof Heightmap Completion via Diffusion Model

This repository contains the implementation of a master thesis "Structure Guided Roof Heightmap Completion via Diffusion Model"

## Overview

The project implements a two-stage approach for roof heightmap completion:

1. **Initial Roof Completion** (`first_roofcompletion/`)
   - Multiple modules: BCE, roof completion, and roofline prediction.
   - Diffusion-based completion with roofline predicting.

2. **Patch Upsampling with Semantics** (`second_patchupsampling_withsem/`)
   - Semantic-guided patch upsampling for enhanced detail
   - Includes semantic encoder for structure understanding


## Directory Structure

- `first_roofcompletion/` - Completion models and training scripts
- `second_patchupsampling_withsem/` - Semantic-guided upsampling pipeline
- `interpolation/` - Interpolation methods for comparison
- `datasets/` - Training data (full-sized, small-sized) and evaluation benchmarks
- `tools/` - Utility scripts for data processing, including edge line rendering sripts, file list generation, heightmap and foorprint generation

## Requirements

```bash
# Create conda environment
conda env create -f environment.yml
conda activate roofcompletion
```

## Usage

### Data Preparation

1. **Extract Datasets**
   Extract the datasets to the project root directory


2. **Generate File Lists**
   Use the file ordering tool to generate relative path file lists:
   ```bash
   cd tools/fileorder
   python flist.py
   ```
   This will generate `.flist` files containing relative paths to your data files.
   The `.flist` format contains relative paths to images in the following format:
   ```
   relative/path/to/image1.png
   relative/path/to/image2.png
   relative/path/to/image3.png
   ```
   Each line contains the relative path from the project root to the image file.

3. **Update Configuration Paths**
   Modify the JSON/yml configuration files to match your dataset paths:

   - `first_roofcompletion/mom/mom.yaml`: Update the following paths:
     ```yaml
     datasets:
       train:
         which_dataset:
           args:
             data_root: "./dataset/fix128/heightmap.flist" #gt
             corrupted_root: "./dataset/fix128/corrupt.flist" 
             footprint_root: "./dataset/fix128/footprint.flist"
             roofline_root: "./dataset/fix128/edge.flist"
     ```
   - `second_patchupsampling_withsem/config/patch.json`: Update the dataset paths accordingly
   
   Update the paths according to your extracted dataset structure and generated flist files.



### 1. First Stage - Roof Completion
```bash
cd first_roofcompletion/mom
python run.py -p train -c mom.yaml
```

### 2. Second Stage - Patch Upsampling
```bash
cd second_patchupsampling_withsem
python run.py -p train -c \config\patch.json
```

### Classical interpolation methods for comparison
```bash
#completion
cd interpolation/completion_interpolation
python run_final_interpolation.py

#upsampling
cd interpolation/upsampling_interpolation
python run_resize_interpolation.py
```

## Models

Pre-trained models and complete training code are available at:
[Google Drive](https://drive.google.com/drive/folders/1wl0aOA0BR8sRZbQnBxnTi5QOdoewt8XJ?usp=sharing)

The Google Drive folder contains:
- Complete training code and configurations
- Pre-trained model weights
- Full training datasets


Pre-trained models are available:
- `first_roofcompletion/mom/2_Network.pth` - BCE module
- `first_roofcompletion/roof/2_Network.pth` - Roof completion model
- `first_roofcompletion/roofline/5_Network.pth` - Roofline prediction model
- `second_patchupsampling_withsem/30_Network.pth` - Upsampling model
- `second_patchupsampling_withsem/semencoder/checkpoint_epoch_15.pth` -Semantic encoder

