#!/usr/bin/env python3
"""
AutoEncoder快速开始脚本
用于测试模型架构和数据加载是否正常工作
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from model import AutoEncoder, count_parameters
from dataset import DepthImageDataset, create_dataloader


def test_model():
    """测试模型架构"""
    print("=" * 50)
    print("测试模型架构...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = AutoEncoder(input_channels=1, latent_dim=768, output_size=128)
    model = model.to(device)
    
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 128, 128).to(device)
    
    print(f"输入形状: {test_input.shape}")
    
    # 完整前向传播
    with torch.no_grad():
        reconstructed, latent = model(test_input)
        print(f"重建图像形状: {reconstructed.shape}")
        print(f"潜在向量形状: {latent.shape}")
        
        # 测试单独的编码和解码
        encoded = model.encode(test_input)
        decoded = model.decode(encoded)
        print(f"编码形状: {encoded.shape}")
        print(f"解码形状: {decoded.shape}")
    
    print("✅ 模型架构测试通过!")
    return True


def test_dataset(data_dir: str = None):
    """测试数据集加载"""
    print("=" * 50)
    print("测试数据集...")
    
    if data_dir is None or not Path(data_dir).exists():
        print("❌ 未提供有效的数据目录，跳过数据集测试")
        print("请使用 --data_dir 参数指定数据目录")
        return False
    
    try:
        # 创建数据集
        dataset = DepthImageDataset(
            data_dir=data_dir,
            image_size=128,
            augment=True,
            height_range=10.0
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 数据集为空，请检查数据目录")
            return False
        
        # 测试获取单个样本
        sample = dataset[0]
        print(f"样本形状: {sample.shape}")
        print(f"样本数据类型: {sample.dtype}")
        print(f"样本数值范围: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # 测试数据加载器
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            augment=True
        )
        
        print(f"数据加载器批次数: {len(dataloader)}")
        
        # 测试获取一个批次
        batch = next(iter(dataloader))
        print(f"批次形状: {batch.shape}")
        print(f"批次数值范围: [{batch.min():.3f}, {batch.max():.3f}]")
        
        print("✅ 数据集测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        return False


def test_training_step(data_dir: str = None):
    """测试一个训练步骤"""
    print("=" * 50)
    print("测试训练步骤...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = AutoEncoder(input_channels=1, latent_dim=768, output_size=128)
    model = model.to(device)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    if data_dir and Path(data_dir).exists():
        # 使用真实数据
        try:
            dataloader = create_dataloader(
                data_dir=data_dir,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                augment=False
            )
            data = next(iter(dataloader))
        except:
            # 如果真实数据加载失败，使用随机数据
            data = torch.randn(2, 1, 128, 128)
            print("使用随机数据进行测试")
    else:
        # 使用随机数据
        data = torch.randn(2, 1, 128, 128)
        print("使用随机数据进行测试")
    
    data = data.to(device)
    
    # 模拟训练步骤
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    reconstructed, latent = model(data)
    
    # 计算损失
    loss = criterion(reconstructed, data)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"训练损失: {loss.item():.6f}")
    print(f"梯度范数: {torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')):.6f}")
    
    print("✅ 训练步骤测试通过!")
    return True


def create_dummy_data(output_dir: str = "./dummy_data", num_images: int = 10):
    """创建虚拟数据用于测试"""
    print("=" * 50)
    print(f"创建虚拟数据到 {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    
    for i in range(num_images):
        # 创建随机的uint16深度图
        # 模拟深度图的特点：大部分区域有值，某些区域为0
        height, width = 128, 128
        
        # 创建基础深度图
        depth = np.random.randint(1000, 50000, (height, width), dtype=np.uint16)
        
        # 添加一些无效区域（值为0）
        mask = np.random.random((height, width)) > 0.2
        depth = depth * mask.astype(np.uint16)
        
        # 添加一些形状特征
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 < (30 + i * 2) ** 2
        depth[circle_mask] = depth[circle_mask] + 10000
        
        # 保存为PNG文件
        image = Image.fromarray(depth, mode='I;16')
        image.save(output_path / f"dummy_depth_{i:03d}.png")
    
    print(f"✅ 创建了 {num_images} 张虚拟深度图")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='AutoEncoder快速开始测试')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--create_dummy', action='store_true', help='创建虚拟数据')
    parser.add_argument('--dummy_dir', type=str, default='./dummy_data', help='虚拟数据输出目录')
    parser.add_argument('--num_dummy', type=int, default=10, help='虚拟数据数量')
    
    args = parser.parse_args()
    
    print("🚀 AutoEncoder 快速开始测试")
    print("=" * 50)
    
    # 创建虚拟数据
    if args.create_dummy:
        dummy_data_dir = create_dummy_data(args.dummy_dir, args.num_dummy)
        if not args.data_dir:
            args.data_dir = dummy_data_dir
    
    # 运行测试
    tests_passed = 0
    total_tests = 3
    
    # 测试模型
    if test_model():
        tests_passed += 1
    
    # 测试数据集
    if test_dataset(args.data_dir):
        tests_passed += 1
    
    # 测试训练步骤
    if test_training_step(args.data_dir):
        tests_passed += 1
    
    # 总结
    print("=" * 50)
    print(f"测试结果: {tests_passed}/{total_tests} 通过")
    
    if tests_passed == total_tests:
        print("🎉 所有测试通过！可以开始正式训练了")
        print("\n下一步:")
        if args.data_dir:
            print(f"python train.py --data_dir {args.data_dir} --batch_size 8 --num_epochs 10")
        else:
            print("python train.py --data_dir /path/to/your/data --batch_size 32 --num_epochs 100")
    else:
        print("❌ 部分测试失败，请检查配置")
    
    print("=" * 50)


if __name__ == '__main__':
    main() 