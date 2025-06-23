#!/usr/bin/env python3
"""
AutoEncoder使用示例
演示如何使用训练好的模型进行推理和编码
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from model import AutoEncoder
from dataset import DepthImageDataset
from evaluate import ModelEvaluator


def example_1_load_and_inference():
    """示例1: 加载模型并进行推理"""
    print("=" * 60)
    print("示例1: 加载模型并进行推理")
    print("=" * 60)
    
    # 模型路径 - 请替换为你的实际模型路径
    model_path = "outputs/autoencoder_20250531_225801/best_model_weights.pth"
    
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或者修改模型路径")
        return
    
    # 创建评估器
    try:
        evaluator = ModelEvaluator(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 创建一个示例图像（随机数据）
    example_image = torch.randn(1, 1, 128, 128)
    print(f"输入图像形状: {example_image.shape}")
    
    # 编码到潜在空间
    latent_vector = evaluator.encode_image(example_image)
    print(f"潜在向量形状: {latent_vector.shape}")
    print(f"潜在向量统计: mean={latent_vector.mean():.4f}, std={latent_vector.std():.4f}")
    
    # 从潜在向量重建图像
    reconstructed = evaluator.decode_latent(latent_vector)
    print(f"重建图像形状: {reconstructed.shape}")
    
    # 计算重建误差
    mse_error = torch.nn.functional.mse_loss(reconstructed, example_image)
    print(f"重建MSE误差: {mse_error.item():.6f}")
    
    print("✅ 推理测试完成")


def example_2_process_real_image():
    """示例2: 处理真实深度图像"""
    print("=" * 60)
    print("示例2: 处理真实深度图像")
    print("=" * 60)
    
    # 检查是否有虚拟数据
    data_dir = "dummy_data"
    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请先运行 python quick_start.py --create_dummy 来创建测试数据")
        return
    
    # 模型路径
    model_path = "outputs/autoencoder_20250531_225801/best_model_weights.pth"
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    try:
        # 加载模型
        evaluator = ModelEvaluator(model_path)
        
        # 创建数据集
        dataset = DepthImageDataset(data_dir, augment=False)
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return
        
        # 获取第一张图像
        image = dataset[0]
        print(f"处理图像形状: {image.shape}")
        
        # 编码和重建
        reconstructed, latent = evaluator.reconstruct_image(image)
        
        print(f"潜在向量维度: {latent.shape[1]}")
        print(f"重建误差: {torch.nn.functional.mse_loss(reconstructed, image.unsqueeze(0)).item():.6f}")
        
        # 可视化结果
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 原图
        axes[0].imshow(image[0].numpy(), cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # 重建图
        axes[1].imshow(reconstructed[0, 0].numpy(), cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        # 差异图
        diff = torch.abs(image[0] - reconstructed[0, 0]).numpy()
        axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title('Difference')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('example_reconstruction.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ 图像处理完成，结果已保存为 'example_reconstruction.png'")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")


def example_3_batch_encoding():
    """示例3: 批量编码图像"""
    print("=" * 60)
    print("示例3: 批量编码图像")
    print("=" * 60)
    
    data_dir = "dummy_data"
    model_path = "outputs/autoencoder_20250531_225801/best_model_weights.pth"
    
    if not Path(data_dir).exists() or not Path(model_path).exists():
        print("❌ 缺少数据或模型文件")
        return
    
    try:
        # 加载模型
        evaluator = ModelEvaluator(model_path)
        
        # 创建数据集和数据加载器
        from dataset import create_dataloader
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            augment=False
        )
        
        print(f"数据集大小: {len(dataloader.dataset)}")
        
        # 批量编码
        all_latents = []
        all_images = []
        
        for batch in dataloader:
            with torch.no_grad():
                latents = evaluator.model.encode(batch.to(evaluator.device))
                all_latents.append(latents.cpu())
                all_images.append(batch)
        
        # 合并所有结果
        all_latents = torch.cat(all_latents, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        print(f"编码完成: {all_latents.shape[0]} 张图像")
        print(f"潜在向量形状: {all_latents.shape}")
        
        # 分析潜在向量
        print(f"潜在空间统计:")
        print(f"  平均值: {all_latents.mean():.4f}")
        print(f"  标准差: {all_latents.std():.4f}")
        print(f"  最小值: {all_latents.min():.4f}")
        print(f"  最大值: {all_latents.max():.4f}")
        
        # 计算图像间的潜在空间距离
        if len(all_latents) >= 2:
            dist = torch.norm(all_latents[0] - all_latents[1])
            print(f"前两张图像的潜在空间距离: {dist.item():.4f}")
        
        print("✅ 批量编码完成")
        
    except Exception as e:
        print(f"❌ 批量编码失败: {e}")


def example_4_custom_model_usage():
    """示例4: 直接使用模型类"""
    print("=" * 60)
    print("示例4: 直接使用模型类")
    print("=" * 60)
    
    try:
        # 创建模型
        model = AutoEncoder(input_channels=1, latent_dim=768, output_size=128)
        
        # 加载权重
        model_path = "outputs/autoencoder_20250531_225801/best_model_weights.pth"
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("✅ 模型权重加载成功")
        else:
            print("⚠️ 模型权重不存在，使用随机初始化的权重")
        
        model.eval()
        
        # 创建测试数据
        test_data = torch.randn(3, 1, 128, 128)
        
        with torch.no_grad():
            # 完整的前向传播
            reconstructed, latent = model(test_data)
            print(f"完整前向传播: {test_data.shape} -> {latent.shape} -> {reconstructed.shape}")
            
            # 只编码
            encoded = model.encode(test_data)
            print(f"仅编码: {test_data.shape} -> {encoded.shape}")
            
            # 只解码
            decoded = model.decode(encoded)
            print(f"仅解码: {encoded.shape} -> {decoded.shape}")
            
            # 验证一致性
            diff = torch.mean(torch.abs(reconstructed - decoded))
            print(f"编码-解码一致性误差: {diff.item():.6f}")
        
        print("✅ 直接模型使用测试完成")
        
    except Exception as e:
        print(f"❌ 直接模型使用失败: {e}")


def main():
    """运行所有示例"""
    print("🚀 AutoEncoder 使用示例")
    print("这些示例展示了如何使用训练好的AutoEncoder模型")
    
    # 运行示例
    example_1_load_and_inference()
    example_2_process_real_image()
    example_3_batch_encoding()
    example_4_custom_model_usage()
    
    print("\n" + "=" * 60)
    print("📋 使用小结:")
    print("1. 使用 ModelEvaluator 类最方便")
    print("2. 支持单张图像和批量处理")
    print("3. 潜在向量可用于下游任务")
    print("4. 可以进行图像重建和质量评估")
    print("=" * 60)


if __name__ == '__main__':
    main() 