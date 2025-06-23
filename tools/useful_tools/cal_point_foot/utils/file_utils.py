import os
from collections import defaultdict
from PIL import Image
import numpy as np
def scan_files(path1,path2,extensions=["las","ply"]):
    """
    Scan all files in the folder and group them by the same name file
    path1: folder path1(las)
    path2: folder path2(ply)
    extensions: file extensions(default=["las","ply"])
    return: defaultdict
    """
    files = defaultdict(lambda: {"ext1": None, "ext2": None})

    for root, dirs, filenames in os.walk(path1):
        for filename in filenames:
            name, ext = filename.rsplit(".", 1)
            if ext == extensions[0]:
                files[name]["ext1"] = os.path.join(root, filename)
    
    for root, dirs, filenames in os.walk(path2):
        for filename in filenames:
            name, ext = filename.rsplit(".", 1)
            if ext == extensions[1]:
                files[name]["ext2"] = os.path.join(root, filename)
    
    return {k: v for k, v in files.items() if v["ext1"] and v["ext2"]}

def write_to_png(image, output_file, mode):
    """
    write numpy array to png file
    image: numpy array
    output_file: output file path
    mode: image mode(default='L')(L: 8-bit pixels, black and white, I: 32-bit signed integer pixels, I;16: 16-bit signed integer pixels)
    """
    #check data type of image
    if image.dtype == "uint16" and mode == "L":
        #image = image.astype("uint8")
        max_range = np.max(image)-np.min(image)
        image = (image - np.min(image)) / max_range * 255
        #image = (image / 65535.0 * 255).astype("uint8")
        image = image.astype("uint8")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    image = Image.fromarray(image, mode=mode)
    image.save(output_file)
    #print(f"Saved projection image as {output_file}")