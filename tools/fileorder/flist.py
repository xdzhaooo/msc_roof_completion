import os


def listdir(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path)
        else:
            with open(path + '.flist', 'a') as f:
                #f.write(file + '\n')
                folder = path.split('/')[-1]
                print(folder)
                f.write(os.path.join(r"./dataset/fix128",folder, file) + '\n')

listdir("dataset/fix128")

#rename all files in the folder by removing "edges_"  in the file name
# import os

# path = "dataset/fix_resolution128/"

# for folder in os.listdir(path):
#     folder_path = os.path.join(path, folder)
#     print(folder_path)
#     if not os.path.isdir(folder_path):  # Ensure it's a folder
#         continue
#     for file in os.listdir(folder_path):
#         if "edges_" in file:  # Only rename files that contain "edges_"
#             file_path = os.path.join(folder_path, file)
#             new_file_path = os.path.join(folder_path, file.replace("edges_", ""))
            
#             os.rename(file_path, new_file_path)  # Rename the file
#             print(f'Renamed: {file_path} -> {new_file_path}')
