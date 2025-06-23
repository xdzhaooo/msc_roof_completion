# Read all subfiles (png format) in the current folder and copy them to another folder
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import random
import math

def get_file_list(file_dir):
    L={}
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png' and "multi_resolution" not in root:
                if "corrupt" in root:
                    L["corrupt"].append(os.path.join(root, file)) if "corrupt" in L else L.update({"corrupt":[os.path.join(root, file)]})
                if "heightmap" in root:
                    L["heightmap"].append(os.path.join(root, file)) if "heightmap" in L else L.update({"heightmap":[os.path.join(root, file)]})
                if "footprint" in root:
                    L["footprint"].append(os.path.join(root, file)) if "footprint" in L else L.update({"footprint":[os.path.join(root, file)]})

    return L

file_dir = r'D:\thesis\progress\2.28'
file_list = get_file_list(file_dir)
for key in file_list:
    print(key, len(file_list[key]))
    dir_path = os.path.join(file_dir,"multi_resolution", key)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for value in file_list[key]:
        shutil.copy(value, dir_path)

