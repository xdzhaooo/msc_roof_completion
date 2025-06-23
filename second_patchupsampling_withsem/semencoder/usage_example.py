#!/usr/bin/env python3
"""
语义编码器使用示例

本文件演示了如何：
1. 从训练好的AutoEncoder中提取Encoder权重
2. 在其他项目中使用SemanticEncoder
3. 各种使用场景和参数控制
"""

import torch
import numpy as np
from pathlib import Path

# 导入我们的模块
from extract_encoder import extract_encoder_weights
from semantic_encoder import SemanticEncoder, create_semantic_encoder


def example_1_extract_encoder_weights():
    """示例1: 从AutoEncoder检查点提取Encoder权重"""
    print("=" * 60)
    print("示例1: 提取Encoder权重")
    print("=" * 60)
    
    # 假设你有一个训练好的AutoEncoder检查点
    autoencoder_checkpoint = "outputs/autoencoder_20250601_134120/best_model.pth"
    
    try:
        # 提取encoder权重
        encoder_weights_path = extract_encoder_weights(
            autoencoder_checkpoint_path=autoencoder_checkpoint,
            output_path="encoder_weights.pth",  # 可选，不指定会自动生成
            input_channels=1,
            latent_dim=768
        )
        print(f"✓ Encoder权重已提取到: {encoder_weights_path}")
        return encoder_weights_path
        
    except FileNotFoundError:
        print("⚠️  检查点文件不存在，跳过此示例")
        print("   请确保有有效的AutoEncoder检查点文件")
        return None
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return None


def example_2_basic_usage(encoder_weights_path: str = None):
    """示例2: 基本使用方法"""
    print("\n" + "=" * 60)
    print("示例2: 基本使用方法")
    print("=" * 60)
    
    # 方法1: 直接创建（不加载预训练权重）
    print("\n--- 方法1: 创建随机初始化的encoder ---")
    encoder = SemanticEncoder(
        input_channels=1,
        latent_dim=768,
        freeze_weights=False
    )
    
    # 方法2: 创建并加载预训练权重
    if encoder_weights_path and Path(encoder_weights_path).exists():
        print("\n--- 方法2: 加载预训练权重 ---")
        encoder_pretrained = SemanticEncoder(
            input_channels=1,
            latent_dim=768,
            pretrained_path=encoder_weights_path,
            freeze_weights=False
        )
    else:
        print("\n--- 方法2: 跳过（无预训练权重文件） ---")
        encoder_pretrained = encoder
    
    # 方法3: 使用便捷函数
    print("\n--- 方法3: 使用便捷函数 ---")
    if encoder_weights_path and Path(encoder_weights_path).exists():
        encoder_easy = create_semantic_encoder(
            pretrained_path=encoder_weights_path,
            freeze_weights=True  # 冻结权重用于特征提取
        )
    else:
        print("跳过便捷函数示例（无预训练权重文件）")
        encoder_easy = encoder
    
    return encoder_pretrained


def example_3_inference(encoder: SemanticEncoder):
    """示例3: 推理使用"""
    print("\n" + "=" * 60)
    print("示例3: 推理使用")
    print("=" * 60)
    
    # 准备测试数据
    batch_size = 4
    height, width = 128, 128
    
    # 方法1: PyTorch tensor输入
    print("\n--- 方法1: PyTorch Tensor推理 ---")
    tensor_input = torch.randn(batch_size, 1, height, width)
    
    with torch.no_grad():
        embeddings = encoder.encode(tensor_input)
    
    print(f"输入形状: {tensor_input.shape}")
    print(f"输出形状: {embeddings.shape}")
    print(f"输出数据类型: {embeddings.dtype}")
    
    # 方法2: NumPy数组输入
    print("\n--- 方法2: NumPy数组推理 ---")
    numpy_input = np.random.randn(batch_size, height, width)  # [N, H, W]
    
    embeddings_numpy = encoder.encode_numpy(
        numpy_input, 
        batch_size=2,  # 分批处理
        normalize=True  # 自动归一化
    )
    
    print(f"NumPy输入形状: {numpy_input.shape}")
    print(f"输出形状: {embeddings_numpy.shape}")
    print(f"输出数据类型: {embeddings_numpy.dtype}")
    
    # 方法3: 大批量推理
    print("\n--- 方法3: 大批量推理 ---")
    large_batch = torch.randn(100, 1, height, width)
    
    large_embeddings = encoder.encode_batch(
        large_batch,
        batch_size=16  # 分批处理避免内存溢出
    )
    
    print(f"大批量输入形状: {large_batch.shape}")
    print(f"输出形状: {large_embeddings.shape}")


def example_4_parameter_control(encoder: SemanticEncoder):
    """示例4: 参数控制（冻结/解冻）"""
    print("\n" + "=" * 60)
    print("示例4: 参数控制")
    print("=" * 60)
    
    # 查看初始状态
    print("--- 初始状态 ---")
    info = encoder.get_model_info()
    print(f"总参数: {info['total_parameters']:,}")
    print(f"可训练参数: {info['trainable_parameters']:,}")
    
    # 冻结所有参数
    print("\n--- 冻结所有参数 ---")
    encoder.freeze_parameters(True)
    print(f"冻结后可训练参数: {encoder.count_parameters(only_trainable=True):,}")
    
    # 解冻所有参数
    print("\n--- 解冻所有参数 ---")
    encoder.freeze_parameters(False)
    print(f"解冻后可训练参数: {encoder.count_parameters(only_trainable=True):,}")
    
    # 冻结特定层
    print("\n--- 冻结特定层 ---")
    encoder.freeze_layers(['stem', 'layer1', 'layer2'])
    print(f"部分冻结后可训练参数: {encoder.count_parameters(only_trainable=True):,}")
    
    # 显示详细信息
    print(f"\n--- 模型详细信息 ---")
    print(encoder)


def example_5_practical_usage():
    """示例5: 实际应用场景"""
    print("\n" + "=" * 60)
    print("示例5: 实际应用场景")
    print("=" * 60)
    
    # 场景1: 作为预训练特征提取器
    print("\n--- 场景1: 预训练特征提取器 ---")
    feature_extractor = SemanticEncoder(
        input_channels=1,
        latent_dim=768,
        freeze_weights=True  # 冻结权重，只用于特征提取
    )
    
    # 处理一批图像数据
    images = torch.randn(10, 1, 128, 128)
    features = feature_extractor.encode_batch(images, batch_size=5)
    print(f"提取了 {features.shape[0]} 个样本的特征，每个特征维度: {features.shape[1]}")
    
    # 场景2: 作为下游任务的backbone
    print("\n--- 场景2: 下游任务backbone ---")
    class DownstreamModel(torch.nn.Module):
        def __init__(self, encoder_path: str = None):
            super().__init__()
            
            # 加载预训练的encoder
            self.backbone = SemanticEncoder(
                input_channels=1,
                latent_dim=768,
                pretrained_path=encoder_path,
                freeze_weights=False  # 允许微调
            )
            
            # 冻结早期层，只微调后面的层
            self.backbone.freeze_layers(['stem', 'layer1'])
            
            # 添加任务特定的头部
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, 10)  # 假设10个类别
            )
        
        def forward(self, x):
            features = self.backbone.encode(x)
            return self.classifier(features)
    
    # 创建下游模型
    downstream_model = DownstreamModel()
    
    # 测试
    test_input = torch.randn(3, 1, 128, 128)
    output = downstream_model(test_input)
    print(f"下游任务输出形状: {output.shape}")
    
    # 场景3: 相似度计算
    print("\n--- 场景3: 相似度计算 ---")
    encoder = SemanticEncoder(input_channels=1, latent_dim=768)
    
    # 计算两个图像的相似度
    img1 = torch.randn(1, 1, 128, 128)
    img2 = torch.randn(1, 1, 128, 128)
    
    with torch.no_grad():
        feat1 = encoder.encode(img1)
        feat2 = encoder.encode(img2)
        
        # 余弦相似度
        similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
        print(f"图像相似度: {similarity.item():.4f}")


def example_6_model_saving_loading():
    """示例6: 模型保存和加载"""
    print("\n" + "=" * 60)
    print("示例6: 模型保存和加载")
    print("=" * 60)
    
    # 创建一个encoder
    encoder = SemanticEncoder(input_channels=1, latent_dim=768)
    
    # 保存权重
    save_path = "my_encoder_weights.pth"
    encoder.save_encoder_weights(save_path)
    
    # 创建新的encoder并加载权重
    new_encoder = SemanticEncoder(
        input_channels=1,
        latent_dim=768,
        pretrained_path=save_path
    )
    
    # 验证加载是否成功
    test_input = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        output1 = encoder.encode(test_input)
        output2 = new_encoder.encode(test_input)
        
        # 检查输出是否相同
        diff = torch.abs(output1 - output2).max()
        print(f"加载前后输出差异: {diff.item():.8f}")
        
        if diff < 1e-6:
            print("✓ 权重加载成功！")
        else:
            print("❌ 权重加载可能有问题")


def main():
    """主函数：运行所有示例"""
    print("🚀 语义编码器使用示例")
    print("本示例将演示如何使用语义编码器的各种功能")
    
    # 示例1: 提取encoder权重
    encoder_weights_path = example_1_extract_encoder_weights()
    
    # 示例2: 基本使用
    encoder = example_2_basic_usage(encoder_weights_path)
    
    # 示例3: 推理使用
    example_3_inference(encoder)
    
    # 示例4: 参数控制
    example_4_parameter_control(encoder)
    
    # 示例5: 实际应用场景
    example_5_practical_usage()
    
    # 示例6: 保存和加载
    example_6_model_saving_loading()
    
    print("\n" + "=" * 60)
    print("✓ 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main() 