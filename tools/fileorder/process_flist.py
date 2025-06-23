import os
import glob

# 要删除的文件名列表（不包含路径，只有文件名）
files_to_remove = [
    "BAG_0599100010060970_ahn4.png",
    "BAG_0599100010052128_ahn4.png",
    "BAG_0599100010030118_ahn4.png",
    "BAG_0599100010006937_ahn3.png",
    "BAG_0599100000755885_ahn4.png", 
    "BAG_0599100000670893_ahn4.png", 
    "BAG_0599100000654952_ahn4.png", 
    "BAG_0599100000624537_ahn3.png", 
    "BAG_0599100000614400_ahn3.png",
    "BAG_0599100000379118_ahn4.png",
    "BAG_0599100000318777_ahn4.png",
    "BAG_0599100000212561_ahn3.png",
    "BAG_0599100000051700_ahn4.png",
    "bag_0518100001637409_ahn3.png",
    "bag_0518100000321964_ahn3.png", 
    "bag_0518100000286691_ahn4.png", 
    "bag_0518100000215544_ahn3.png"
]

# 找到所有的 .flist 文件
flist_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.flist') and not file.endswith('~small.flist'):
            flist_files.append(os.path.join(root, file))

print(f"找到 {len(flist_files)} 个 .flist 文件:")
for file in flist_files:
    print(f"  {file}")

# 跟踪找到的文件
found_files = set()
all_removed_files = []

# 处理每个 .flist 文件
for flist_file in flist_files:
    print(f"\n处理文件: {flist_file}")
    
    # 读取原文件内容
    with open(flist_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤掉要删除的文件名行
    filtered_lines = []
    removed_count = 0
    removed_in_this_file = []
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped:
            # 提取文件名部分（去掉路径）
            filename = os.path.basename(line_stripped)
            
            # 为了能匹配 footprint 文件, 我们创建一个用于比较的版本
            # 例如: '..._footprint.png' -> '....png'
            compare_filename = filename.replace('_footprint', '')
            
            # 检查是否在要删除的列表中
            if compare_filename in files_to_remove:
                print(f"  删除行: {line_stripped}")
                removed_count += 1
                removed_in_this_file.append(filename)
                found_files.add(compare_filename) # 使用列表中的名字来追踪
                all_removed_files.append((flist_file, filename))
            else:
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)
    
    # 生成新文件名
    directory = os.path.dirname(flist_file)
    filename = os.path.basename(flist_file)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}~small{ext}"
    new_filepath = os.path.join(directory, new_filename)
    
    # 写入新文件
    with open(new_filepath, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)
    
    print(f"  原文件行数: {len(lines)}")
    print(f"  删除行数: {removed_count}")
    print(f"  新文件行数: {len(filtered_lines)}")
    print(f"  新文件保存为: {new_filepath}")

# 检查哪些文件没有在任何 .flist 中找到
not_found_files = set(files_to_remove) - found_files

print(f"\n\n=== 删除统计报告 ===")
print(f"要删除的文件总数: {len(files_to_remove)}")
print(f"实际找到并删除的文件数: {len(found_files)}")
print(f"未找到的文件数: {len(not_found_files)}")

if found_files:
    print(f"\n找到并删除的文件:")
    for filename in sorted(found_files):
        print(f"  ✓ {filename}")

if not_found_files:
    print(f"\n⚠️ 未在任何 .flist 文件中找到的文件:")
    for filename in sorted(not_found_files):
        print(f"  ✗ {filename}")

print(f"\n=== 各文件删除详情 ===")
files_by_flist = {}
for flist_file, filename in all_removed_files:
    if flist_file not in files_by_flist:
        files_by_flist[flist_file] = []
    files_by_flist[flist_file].append(filename)

for flist_file in sorted(files_by_flist.keys()):
    print(f"\n{flist_file} (删除了 {len(files_by_flist[flist_file])} 个文件):")
    for filename in sorted(files_by_flist[flist_file]):
        print(f"  - {filename}")

print("\n所有文件处理完成！") 