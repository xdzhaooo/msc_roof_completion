# 语义编码器工具包

本工具包提供了从训练好的AutoEncoder中提取Encoder部分，并创建独立可用的语义编码器的完整解决方案。

## 📁 文件结构

```
├── extract_encoder.py      # 从AutoEncoder中提取Encoder权重
├── semantic_encoder.py     # 独立的语义编码器接口
├── usage_example.py        # 完整使用示例
├── README_semantic_encoder.md  # 本文档
├── model.py               # 原始模型定义
└── train.py              # 原始训练脚本
```

## 🚀 快速开始

### 1. 提取Encoder权重

从训练好的AutoEncoder检查点中提取Encoder部分：

```bash
# 基本使用
python extract_encoder.py path/to/autoencoder_checkpoint.pth

# 指定输出路径
python extract_encoder.py path/to/autoencoder_checkpoint.pth --output encoder_weights.pth

# 验证提取的权重
python extract_encoder.py path/to/autoencoder_checkpoint.pth --verify
```

### 2. 使用语义编码器

```python
from semantic_encoder import SemanticEncoder, create_semantic_encoder

# 方法1: 直接创建
encoder = SemanticEncoder(
    input_channels=1,
    latent_dim=768,
    pretrained_path="encoder_weights.pth",
    freeze_weights=False
)

# 方法2: 使用便捷函数
encoder = create_semantic_encoder(
    pretrained_path="encoder_weights.pth",
    freeze_weights=True  # 用于特征提取
)

# 推理
import torch
x = torch.randn(4, 1, 128, 128)  # [B, C, H, W]
embeddings = encoder.encode(x)   # [B, 768]
```

## 📚 详细功能

### extract_encoder.py

**功能**：从完整的AutoEncoder检查点中提取Encoder权重

**参数**：
- `checkpoint_path`: AutoEncoder检查点文件路径
- `--output`: 输出的encoder权重文件路径（可选）
- `--input_channels`: 输入通道数（默认：1）
- `--latent_dim`: 潜在向量维度（默认：768）
- `--verify`: 验证提取的权重

**输出格式**：
```python
{
    'encoder_state_dict': dict,  # Encoder权重
    'model_config': {
        'input_channels': int,
        'latent_dim': int,
        'model_type': str
    },
    'extracted_from': str,  # 原始文件路径
    'extraction_info': {
        'total_params': int,
        'trainable_params': int,
        'num_weights': int
    }
}
```

### semantic_encoder.py

**主要类**：`SemanticEncoder`

**初始化参数**：
- `input_channels`: 输入通道数（默认：1）
- `latent_dim`: 潜在向量维度（默认：768）
- `pretrained_path`: 预训练权重文件路径
- `freeze_weights`: 是否冻结所有权重
- `device`: 运行设备

**主要方法**：

#### 权重管理
```python
# 加载预训练权重
encoder.load_pretrained_weights("path/to/weights.pth")

# 冻结/解冻所有参数
encoder.freeze_parameters(True)  # 冻结
encoder.freeze_parameters(False) # 解冻

# 冻结特定层
encoder.freeze_layers(['stem', 'layer1', 'layer2'])

# 保存权重
encoder.save_encoder_weights("output_weights.pth")
```

#### 推理方法
```python
# PyTorch Tensor推理
embeddings = encoder.encode(tensor_input)

# NumPy数组推理（自动处理维度和归一化）
embeddings = encoder.encode_numpy(numpy_input, normalize=True)

# 大批量推理（自动分批）
embeddings = encoder.encode_batch(large_tensor, batch_size=16)
```

#### 信息查询
```python
# 参数数量
total_params = encoder.count_parameters()
trainable_params = encoder.count_parameters(only_trainable=True)

# 模型信息
info = encoder.get_model_info()
print(encoder)  # 详细信息
```

## 🔧 使用场景

### 1. 特征提取器

```python
# 冻结权重，用作特征提取器
feature_extractor = SemanticEncoder(
    pretrained_path="encoder_weights.pth",
    freeze_weights=True
)

# 批量提取特征
images = torch.randn(100, 1, 128, 128)
features = feature_extractor.encode_batch(images, batch_size=16)
```

### 2. 下游任务的Backbone

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练encoder
        self.backbone = SemanticEncoder(
            pretrained_path="encoder_weights.pth",
            freeze_weights=False
        )
        # 冻结早期层
        self.backbone.freeze_layers(['stem', 'layer1'])
        
        # 任务特定的头部
        self.classifier = torch.nn.Linear(768, num_classes)
    
    def forward(self, x):
        features = self.backbone.encode(x)
        return self.classifier(features)
```

### 3. 相似度计算

```python
encoder = SemanticEncoder(pretrained_path="encoder_weights.pth")

# 计算两个图像的相似度
with torch.no_grad():
    feat1 = encoder.encode(img1)
    feat2 = encoder.encode(img2)
    similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
```

### 4. 数据处理流水线

```python
import numpy as np

# 处理NumPy格式的数据
def process_depth_maps(depth_maps: np.ndarray) -> np.ndarray:
    """
    处理深度图数组
    Args:
        depth_maps: [N, H, W] 或 [N, C, H, W]
    Returns:
        embeddings: [N, 768]
    """
    encoder = SemanticEncoder(
        pretrained_path="encoder_weights.pth",
        freeze_weights=True
    )
    
    return encoder.encode_numpy(
        depth_maps, 
        batch_size=32,
        normalize=True
    )
```

## ⚙️ 配置参数

### 模型配置
- **input_channels**: 输入通道数
  - `1`: 深度图/灰度图
  - `3`: RGB图像
- **latent_dim**: 潜在向量维度
  - 默认：`768`（与训练时保持一致）

### 设备配置
- **device**: 运行设备
  - `"cuda"`: GPU
  - `"cpu"`: CPU
  - `None`: 自动选择

### 权重冻结策略
- **freeze_weights**: 全局冻结开关
- **freeze_layers**: 选择性冻结特定层
  - 常用层名：`["stem", "layer1", "layer2", "layer3", "layer4"]`

## 🔍 调试与验证

### 权重验证
```python
# 检查权重加载是否成功
encoder = SemanticEncoder(pretrained_path="weights.pth")
test_input = torch.randn(1, 1, 128, 128)
output = encoder.encode(test_input)
print(f"输出形状: {output.shape}")  # 应该是 [1, 768]
```

### 参数检查
```python
# 查看参数状态
info = encoder.get_model_info()
print(f"总参数: {info['total_parameters']:,}")
print(f"可训练参数: {info['trainable_parameters']:,}")

# 检查梯度状态
for name, param in encoder.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
```

### 性能测试
```python
import time

# 测试推理速度
encoder.eval()
test_batch = torch.randn(32, 1, 128, 128)

with torch.no_grad():
    start_time = time.time()
    output = encoder.encode(test_batch)
    end_time = time.time()

print(f"批量推理时间: {(end_time - start_time)*1000:.2f} ms")
print(f"单张图像推理时间: {(end_time - start_time)*1000/32:.2f} ms")
```

## 📋 完整示例

参见 `usage_example.py` 文件，包含了所有功能的详细示例：

```bash
python usage_example.py
```

示例包括：
1. 从AutoEncoder提取权重
2. 基本使用方法
3. 推理使用
4. 参数控制
5. 实际应用场景
6. 模型保存和加载

## 🐛 常见问题

### Q1: 权重加载失败
**问题**：`权重文件不存在` 或 `未找到encoder权重`

**解决**：
1. 检查文件路径是否正确
2. 确保使用 `extract_encoder.py` 正确提取了权重
3. 检查原始AutoEncoder检查点是否完整

### Q2: 输入维度不匹配
**问题**：`Expected 4D tensor, got 3D tensor`

**解决**：
```python
# 确保输入是4D张量 [B, C, H, W]
if input_tensor.ndim == 3:
    input_tensor = input_tensor.unsqueeze(1)  # 添加通道维度
```

### Q3: 内存不足
**问题**：GPU内存不足

**解决**：
```python
# 使用批量推理
embeddings = encoder.encode_batch(large_input, batch_size=16)

# 或者使用CPU
encoder = SemanticEncoder(device='cpu')
```

### Q4: 参数冻结不生效
**问题**：参数仍然在更新

**解决**：
```python
# 检查参数状态
for name, param in encoder.named_parameters():
    if param.requires_grad:
        print(f"未冻结的参数: {name}")

# 强制冻结
encoder.freeze_parameters(True)
```

## 📝 注意事项

1. **版本兼容性**：确保PyTorch版本与训练时一致
2. **设备一致性**：权重文件和模型需要在同一设备上
3. **输入格式**：确保输入数据格式与训练时一致（归一化范围等）
4. **内存管理**：大批量推理时注意内存使用
5. **梯度计算**：推理时记得使用 `torch.no_grad()`

## 🔄 更新日志

- **v1.0**: 基础功能实现
  - 权重提取
  - 基本推理接口
  - 参数冻结控制
  - NumPy兼容性

## 📞 支持

如有问题，请检查：
1. 运行 `usage_example.py` 确认基本功能
2. 查看错误信息和堆栈跟踪
3. 检查输入数据格式和维度 