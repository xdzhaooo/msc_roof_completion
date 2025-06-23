import os
import glob
from collections import defaultdict

# 找到所有的 ~small.flist 文件
small_flist_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('~small.flist'):
            small_flist_files.append(os.path.join(root, file))

print(f"找到 {len(small_flist_files)} 个 ~small.flist 文件:")
for file in small_flist_files:
    print(f"  {file}")

# 读取每个文件的内容，提取文件名序列
file_sequences = {}
for flist_file in small_flist_files:
    with open(flist_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 提取每行的文件名（去掉路径和换行符）
    filenames = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped:
            basename = os.path.basename(line_stripped)
            # 统一移除_footprint后缀进行比较
            compare_name = basename.replace('_footprint.png', '.png')
            filenames.append(compare_name)
    
    file_sequences[flist_file] = filenames
    print(f"\n{flist_file}: {len(filenames)} 个文件")

# 检查所有文件的序列长度是否相同
lengths = [len(seq) for seq in file_sequences.values()]
if len(set(lengths)) == 1:
    print(f"\n✓ 所有文件的行数都相同: {lengths[0]} 行")
else:
    print(f"\n⚠️ 文件行数不同:")
    for file, seq in file_sequences.items():
        print(f"  {file}: {len(seq)} 行")

# 比较所有文件的序列是否相同
print("\n=== 检查文件顺序一致性 ===")
file_list = list(file_sequences.keys())
reference_file = file_list[0]
reference_sequence = file_sequences[reference_file]

all_same = True
different_files = []

for i, compare_file in enumerate(file_list[1:], 1):
    compare_sequence = file_sequences[compare_file]
    
    if compare_sequence == reference_sequence:
        print(f"✓ {compare_file} 与参考文件顺序一致")
    else:
        print(f"✗ {compare_file} 与参考文件顺序不一致")
        all_same = False
        different_files.append(compare_file)
        
        # 找出不同的位置
        differences = []
        min_len = min(len(reference_sequence), len(compare_sequence))
        for j in range(min_len):
            if reference_sequence[j] != compare_sequence[j]:
                differences.append(j)
        
        if len(reference_sequence) != len(compare_sequence):
            print(f"    长度不同: 参考文件 {len(reference_sequence)} vs {compare_file} {len(compare_sequence)}")
        
        if differences:
            print(f"    前10个不同位置: {differences[:10]}")
            for pos in differences[:5]:  # 显示前5个不同的位置
                ref_file = reference_sequence[pos] if pos < len(reference_sequence) else "超出范围"
                cmp_file = compare_sequence[pos] if pos < len(compare_sequence) else "超出范围"
                print(f"      位置 {pos}: 参考='{ref_file}' vs 比较='{cmp_file}'")

if all_same:
    print(f"\n🎉 所有 {len(file_list)} 个文件的顺序完全一致！")
    print(f"参考文件: {reference_file}")
else:
    print(f"\n⚠️ 发现 {len(different_files)} 个文件的顺序与参考文件不一致")
    print(f"参考文件: {reference_file}")
    print("不一致的文件:")
    for file in different_files:
        print(f"  - {file}")

# 额外检查：查看每个目录中相同类型的文件是否一致
print("\n=== 按目录和类型分组检查 ===")
directory_groups = defaultdict(list)
for file in file_list:
    directory = os.path.dirname(file)
    directory_groups[directory].append(file)

for directory, files in directory_groups.items():
    if len(files) > 1:
        print(f"\n目录 {directory} 中的文件:")
        ref_seq = file_sequences[files[0]]
        for file in files:
            if file_sequences[file] == ref_seq:
                print(f"  ✓ {os.path.basename(file)}")
            else:
                print(f"  ✗ {os.path.basename(file)} (与同目录其他文件不一致)") 