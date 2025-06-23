import os
import shutil

file1 = "dataset/fix128/edge"
file2 = "dataset/fix128/edge (2)"

file2_list = []
for root, dirs, files in os.walk(file2):
    for file in files:
        file2_list.append(file)
for root, dirs, files in os.walk(file1):
    for file in files:
        if file not in file2_list:
            print(f"File {file} not found in {file2}")
            #copy to file2
            shutil.copy(os.path.join(root, file), os.path.join(file2, file))
        