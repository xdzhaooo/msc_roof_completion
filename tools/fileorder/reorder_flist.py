import os

def get_basename_order(filepath):
    """从文件中读取行，并返回一个只包含文件名的列表，作为顺序参考。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    basenames = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped:
            # 统一移除_footprint后缀，以便进行无差别的顺序比较
            basenames.append(os.path.basename(line_stripped).replace('_footprint.png', '.png'))
    return basenames

def reorder_file(target_path, reference_order):
    """根据参考顺序重新排列目标文件中的行。"""
    print(f"\n正在处理文件: {target_path}")
    
    with open(target_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 创建从"无后缀"文件名到完整行内容的映射
    line_map = {}
    for line in lines:
        line_stripped = line.strip()
        if line_stripped:
            basename = os.path.basename(line_stripped)
            compare_name = basename.replace('_footprint.png', '.png')
            line_map[compare_name] = line

    # 根据参考顺序构建新的行列表
    new_lines = []
    reordered_count = 0
    for basename in reference_order:
        if basename in line_map:
            new_lines.append(line_map[basename])
            reordered_count += 1
        else:
            print(f"  - 警告: 在 {target_path} 中未找到 {basename}，该行将被跳过。")
    
    # 覆盖原文件
    with open(target_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
        
    print(f"  完成处理: {reordered_count} 行已按新顺序写入 {target_path}")

# --- 主程序 ---
# 1. 设置参考文件和目标文件
reference_file = 's80i80/roof_footprint~small.flist'
files_to_reorder = [
    '8030/0~small.flist',
    '8080/0~small.flist',
    's80i30/roof_footprint~small.flist',
    's80i30/roof_gt~small.flist',
    's80i30/roof_img~small.flist',
    # 's80i80/roof_footprint~small.flist', # 自身是参考，不需要重排
    's80i80/roof_gt~small.flist',
    's80i80/roof_img~small.flist'
]

print(f"将使用 '{reference_file}' 的顺序作为最终参考标准。")

# 2. 获取参考顺序
try:
    reference_order = get_basename_order(reference_file)
    print(f"参考顺序已加载，包含 {len(reference_order)} 个文件。")
except FileNotFoundError:
    print(f"错误：参考文件 '{reference_file}' 未找到。")
    exit()

# 3. 重新排序所有目标文件
for file_path in files_to_reorder:
    try:
        reorder_file(file_path, reference_order)
    except FileNotFoundError:
        print(f"\n错误：目标文件 '{file_path}' 未找到，跳过。")

print("\n所有文件已按统一标准重新排序！") 