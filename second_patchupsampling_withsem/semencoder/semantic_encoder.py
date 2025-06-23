#!/usr/bin/env python3
"""
语义编码器接口 - 独立的Encoder模块，用于在其他项目中使用
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Union, Optional, Tuple, List
from pathlib import Path
import warnings

from .model import EnhancedResNet50Encoder


class SemanticEncoder(nn.Module):
    """
    语义编码器接口类
    
    提供了完整的encoder功能，包括权重加载、推理、参数冻结控制等
    可以在其他项目中独立使用
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 latent_dim: int = 768,
                 pretrained_path: Optional[str] = None,
                 freeze_weights: bool = False,
                 device: Optional[Union[str, torch.device]] = None):
        """
        初始化语义编码器
        
        Args:
            input_channels: 输入通道数 (默认: 1，适用于深度图/灰度图)
            latent_dim: 潜在向量维度 (默认: 768)
            pretrained_path: 预训练权重文件路径
            freeze_weights: 是否冻结所有权重 (默认: False)
            device: 运行设备，如果为None则自动选择
        """
        super(SemanticEncoder, self).__init__()
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 模型配置
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.freeze_weights = freeze_weights
        
        # 创建encoder模型
        self.encoder = EnhancedResNet50Encoder(
            input_channels=input_channels,
            latent_dim=latent_dim
        )
        
        # 加载预训练权重
        self.pretrained_path = None
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)
        
        # 移动到指定设备
        self.to(self.device)
        
        # 设置权重冻结
        if freeze_weights:
            self.freeze_parameters()
        
        print(f"✓ SemanticEncoder初始化完成")
        print(f"  - 设备: {self.device}")
        print(f"  - 输入通道: {input_channels}")
        print(f"  - 潜在维度: {latent_dim}")
        print(f"  - 参数数量: {self.count_parameters():,}")
        print(f"  - 权重冻结: {freeze_weights}")
        if pretrained_path:
            print(f"  - 预训练权重: {pretrained_path}")
    
    def load_pretrained_weights(self, pretrained_path: str) -> bool:
        """
        加载预训练权重
        
        Args:
            pretrained_path: 权重文件路径
            
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"权重文件不存在: {pretrained_path}")
        
        try:
            print(f"正在加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 检查权重格式
            if 'encoder_state_dict' in checkpoint:
                # 从extract_encoder.py提取的格式
                state_dict = checkpoint['encoder_state_dict']
                config = checkpoint.get('model_config', {})
                
                # 验证配置匹配
                if config.get('input_channels') != self.input_channels:
                    warnings.warn(f"输入通道数不匹配: 期望 {self.input_channels}, 实际 {config.get('input_channels')}")
                if config.get('latent_dim') != self.latent_dim:
                    warnings.warn(f"潜在维度不匹配: 期望 {self.latent_dim}, 实际 {config.get('latent_dim')}")
                    
            elif 'model_state_dict' in checkpoint:
                # 完整模型检查点，需要提取encoder部分
                full_state_dict = checkpoint['model_state_dict']
                state_dict = {}
                encoder_prefix = 'encoder.'
                for key, value in full_state_dict.items():
                    if key.startswith(encoder_prefix):
                        new_key = key[len(encoder_prefix):]
                        state_dict[new_key] = value
                        
                if not state_dict:
                    raise ValueError("未找到encoder权重")
            else:
                # 直接的state_dict
                state_dict = checkpoint
            
            # 加载权重
            missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"警告: 缺失的权重键: {missing_keys}")
            if unexpected_keys:
                print(f"警告: 意外的权重键: {unexpected_keys}")
                
            self.pretrained_path = pretrained_path
            print("✓ 预训练权重加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 加载预训练权重失败: {e}")
            return False
    
    def freeze_parameters(self, freeze: bool = True):
        """
        冻结或解冻模型参数
        
        Args:
            freeze: True为冻结，False为解冻
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        
        self.freeze_weights = freeze
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder.parameters())
        
        status = "冻结" if freeze else "解冻"
        print(f"✓ 参数已{status}: {trainable_params:,}/{total_params:,} 可训练")
    
    def freeze_layers(self, layer_names: List[str]):
        """
        冻结指定的层
        
        Args:
            layer_names: 要冻结的层名称列表
        """
        frozen_count = 0
        for name, param in self.encoder.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break
        
        print(f"✓ 已冻结 {frozen_count} 个参数 (指定层: {layer_names})")
    
    def count_parameters(self, only_trainable: bool = False) -> int:
        """
        计算参数数量
        
        Args:
            only_trainable: 是否只计算可训练参数
            
        Returns:
            参数数量
        """
        if only_trainable:
            return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.encoder.parameters())
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        total_params = self.count_parameters(only_trainable=False)
        trainable_params = self.count_parameters(only_trainable=True)
        
        return {
            'model_type': 'EnhancedResNet50Encoder',
            'input_channels': self.input_channels,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen': self.freeze_weights,
            'device': str(self.device),
            'pretrained_path': self.pretrained_path
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            潜在表示 [B, latent_dim]
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        return self.encoder(x)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入 (forward的别名)
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            潜在表示 [B, latent_dim]
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        return self.forward(x)
    
    @torch.no_grad()
    def encode_batch(self, 
                    x: torch.Tensor,
                    batch_size: Optional[int] = None) -> torch.Tensor:
        """
        批量编码（用于大数据量推理）
        
        Args:
            x: 输入张量 [N, C, H, W]
            batch_size: 批处理大小，如果为None则使用完整批次
            
        Returns:
            潜在表示 [N, latent_dim]
        """
        self.eval()
        
        if batch_size is None or x.size(0) <= batch_size:
            # 直接处理
            x = x.to(self.device)
            return self.encode(x)
        
        # 分批处理
        results = []
        for i in range(0, x.size(0), batch_size):
            batch = x[i:i+batch_size].to(self.device)
            batch_result = self.encode(batch)
            results.append(batch_result.cpu())
        
        return torch.cat(results, dim=0)
    
    @torch.no_grad()
    def encode_numpy(self, 
                    x: np.ndarray,
                    batch_size: Optional[int] = None,
                    normalize: bool = True) -> np.ndarray:
        """
        编码numpy数组
        
        Args:
            x: 输入数组 [N, C, H, W] 或 [N, H, W]
            batch_size: 批处理大小
            normalize: 是否归一化到[-1, 1]
            
        Returns:
            潜在表示 [N, latent_dim]
        """
        # 处理输入维度
        if x.ndim == 3:
            x = x[:, None, :, :]  # [N, H, W] -> [N, 1, H, W]
        elif x.ndim == 4:
            pass  # [N, C, H, W]
        else:
            raise ValueError(f"不支持的输入维度: {x.ndim}")
        
        # 转换为tensor
        x_tensor = torch.from_numpy(x).float()
        
        # 归一化
        if normalize:
            x_tensor = (x_tensor - x_tensor.min()) / (x_tensor.max() - x_tensor.min())
            x_tensor = x_tensor * 2 - 1  # [0, 1] -> [-1, 1]
        
        # 编码
        result = self.encode_batch(x_tensor, batch_size)
        
        return result.cpu().numpy()
    
    def save_encoder_weights(self, save_path: str):
        """
        保存encoder权重
        
        Args:
            save_path: 保存路径
        """
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'model_config': {
                'input_channels': self.input_channels,
                'latent_dim': self.latent_dim,
                'model_type': 'EnhancedResNet50Encoder'
            },
            'model_info': self.get_model_info()
        }
        
        torch.save(checkpoint, save_path)
        print(f"✓ Encoder权重已保存到: {save_path}")
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (f"SemanticEncoder(\n"
                f"  input_channels={info['input_channels']},\n"
                f"  latent_dim={info['latent_dim']},\n"
                f"  parameters={info['total_parameters']:,} "
                f"({info['trainable_parameters']:,} trainable),\n"
                f"  device={info['device']},\n"
                f"  frozen={info['frozen']}\n"
                f")")


def create_semantic_encoder(pretrained_path: str,
                          input_channels: int = 1,
                          latent_dim: int = 768,
                          freeze_weights: bool = False,
                          device: Optional[str] = None) -> SemanticEncoder:
    """
    便捷函数：创建并加载预训练的语义编码器
    
    Args:
        pretrained_path: 预训练权重路径
        input_channels: 输入通道数
        latent_dim: 潜在向量维度
        freeze_weights: 是否冻结权重
        device: 运行设备
        
    Returns:
        配置好的SemanticEncoder实例
    """
    encoder = SemanticEncoder(
        input_channels=input_channels,
        latent_dim=latent_dim,
        pretrained_path=pretrained_path,
        freeze_weights=freeze_weights,
        device=device
    )
    
    return encoder


# 使用示例
if __name__ == "__main__":
    # 示例1: 创建encoder并加载权重
    print("=== 示例1: 基本使用 ===")
    encoder = SemanticEncoder(
        input_channels=1,
        latent_dim=768,
        pretrained_path=None,  # 这里填入实际的权重路径
        freeze_weights=False
    )
    
    # 测试推理
    test_input = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        embedding = encoder.encode(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {embedding.shape}")
    
    # 示例2: 冻结参数
    print("\n=== 示例2: 参数冻结 ===")
    encoder.freeze_parameters(True)
    print(f"冻结后可训练参数: {encoder.count_parameters(only_trainable=True):,}")
    
    # 示例3: 部分层冻结
    print("\n=== 示例3: 部分层冻结 ===")
    encoder.freeze_parameters(False)  # 先解冻所有
    encoder.freeze_layers(['stem', 'layer1'])  # 冻结特定层
    print(f"部分冻结后可训练参数: {encoder.count_parameters(only_trainable=True):,}")
    
    # 示例4: numpy数组编码
    print("\n=== 示例4: numpy数组编码 ===")
    np_input = np.random.randn(3, 128, 128)  # [N, H, W]
    embeddings = encoder.encode_numpy(np_input, batch_size=2)
    print(f"numpy输入形状: {np_input.shape}")
    print(f"编码结果形状: {embeddings.shape}")
    
    # 显示模型信息
    print(f"\n=== 模型信息 ===")
    print(encoder) 