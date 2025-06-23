#!/usr/bin/env python3
"""
从训练好的AutoEncoder权重文件中提取Encoder部分
"""

import torch
import argparse
import os
from pathlib import Path
from model import AutoEncoder, EnhancedResNet50Encoder


def extract_encoder_weights(autoencoder_checkpoint_path: str, output_path: str = None, 
                          input_channels: int = 1, latent_dim: int = 768):
    """
    从AutoEncoder检查点中提取Encoder权重
    
    Args:
        autoencoder_checkpoint_path: AutoEncoder检查点文件路径
        output_path: 输出的encoder权重文件路径，如果为None则自动生成
        input_channels: 输入通道数
        latent_dim: 潜在向量维度
    
    Returns:
        str: 保存的encoder权重文件路径
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(autoencoder_checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {autoencoder_checkpoint_path}")
    
    print(f"正在加载AutoEncoder检查点: {autoencoder_checkpoint_path}")
    
    # 加载检查点
    try:
        checkpoint = torch.load(autoencoder_checkpoint_path, map_location='cpu')
        
        # 检查是否包含模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"检查点信息: Epoch {epoch}")
            if config:
                print(f"训练配置: {config}")
        else:
            # 直接是模型权重
            state_dict = checkpoint
            print("检测到直接的模型权重文件")
            
    except Exception as e:
        raise RuntimeError(f"加载检查点失败: {e}")
    
    # 提取encoder相关的权重
    encoder_state_dict = {}
    encoder_prefix = 'encoder.'
    
    for key, value in state_dict.items():
        if key.startswith(encoder_prefix):
            # 移除'encoder.'前缀
            new_key = key[len(encoder_prefix):]
            encoder_state_dict[new_key] = value
    
    if not encoder_state_dict:
        raise ValueError("在检查点中未找到encoder权重，请检查文件格式")
    
    print(f"成功提取 {len(encoder_state_dict)} 个encoder参数")
    
    # 验证提取的权重是否可以正确加载
    try:
        encoder = EnhancedResNet50Encoder(
            input_channels=input_channels, 
            latent_dim=latent_dim
        )
        encoder.load_state_dict(encoder_state_dict)
        print("✓ 权重验证成功，encoder可以正确加载")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  - 总参数数: {total_params:,}")
        print(f"  - 可训练参数数: {trainable_params:,}")
        
    except Exception as e:
        raise RuntimeError(f"权重验证失败: {e}")
    
    # 生成输出路径
    if output_path is None:
        input_path = Path(autoencoder_checkpoint_path)
        output_path = input_path.parent / f"encoder_weights_{input_path.stem}.pth"
    
    # 保存encoder权重和相关信息
    encoder_checkpoint = {
        'encoder_state_dict': encoder_state_dict,
        'model_config': {
            'input_channels': input_channels,
            'latent_dim': latent_dim,
            'model_type': 'EnhancedResNet50Encoder'
        },
        'extracted_from': autoencoder_checkpoint_path,
        'extraction_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_weights': len(encoder_state_dict)
        }
    }
    
    torch.save(encoder_checkpoint, output_path)
    print(f"✓ Encoder权重已保存到: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='从AutoEncoder检查点中提取Encoder权重')
    parser.add_argument('checkpoint_path', type=str, help='AutoEncoder检查点文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出的encoder权重文件路径')
    parser.add_argument('--input_channels', type=int, default=1, help='输入通道数 (默认: 1)')
    parser.add_argument('--latent_dim', type=int, default=768, help='潜在向量维度 (默认: 768)')
    parser.add_argument('--verify', action='store_true', help='验证提取的权重')
    
    args = parser.parse_args()
    
    try:
        output_path = extract_encoder_weights(
            autoencoder_checkpoint_path=args.checkpoint_path,
            output_path=args.output,
            input_channels=args.input_channels,
            latent_dim=args.latent_dim
        )
        
        if args.verify:
            print("\n=== 验证提取的Encoder权重 ===")
            # 重新加载验证
            encoder_checkpoint = torch.load(output_path, map_location='cpu')
            encoder = EnhancedResNet50Encoder(
                input_channels=args.input_channels,
                latent_dim=args.latent_dim
            )
            encoder.load_state_dict(encoder_checkpoint['encoder_state_dict'])
            
            # 测试前向传播
            test_input = torch.randn(1, args.input_channels, 128, 128)
            with torch.no_grad():
                output = encoder(test_input)
            print(f"✓ 测试前向传播成功: {test_input.shape} -> {output.shape}")
        
        print(f"\n✓ Encoder权重提取完成!")
        print(f"输出文件: {output_path}")
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 