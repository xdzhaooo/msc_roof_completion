# 语义编码器快速开始指南

## 🎯 目标
从训练好的AutoEncoder中提取Encoder部分，创建独立可用的语义编码器。

## 📦 核心文件
1. **extract_encoder.py** - 权重提取工具
2. **semantic_encoder.py** - 语义编码器接口
3. **usage_example.py** - 详细使用示例
4. **test_semantic_encoder.py** - 功能测试

## ⚡ 2分钟快速上手

### 步骤1: 提取Encoder权重
```bash
# 从AutoEncoder检查点提取encoder权重
python extract_encoder.py outputs/autoencoder_20250601_134120/best_model.pth
```

### 步骤2: 使用语义编码器
```python
from semantic_encoder import SemanticEncoder
import torch

# 创建encoder并加载权重
encoder = SemanticEncoder(
    input_channels=1,
    latent_dim=768,
    pretrained_path="encoder_weights_best_model.pth",
    freeze_weights=True  # 冻结权重用于特征提取
)

# 推理
x = torch.randn(4, 1, 128, 128)  # [B, C, H, W]
embeddings = encoder.encode(x)   # [B, 768]
print(f"输入: {x.shape} -> 输出: {embeddings.shape}")
```

## 🔧 常用场景

### 特征提取器
```python
# 冻结所有权重，用作特征提取器
feature_extractor = SemanticEncoder(
    pretrained_path="encoder_weights.pth",
    freeze_weights=True
)

# 批量提取特征
images = torch.randn(100, 1, 128, 128)
features = feature_extractor.encode_batch(images, batch_size=16)
```

### 下游任务backbone
```python
class MyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 预训练encoder作为backbone
        self.backbone = SemanticEncoder(
            pretrained_path="encoder_weights.pth",
            freeze_weights=False  # 允许微调
        )
        # 冻结早期层
        self.backbone.freeze_layers(['stem', 'layer1'])
        
        # 分类头
        self.classifier = torch.nn.Linear(768, 10)
    
    def forward(self, x):
        features = self.backbone.encode(x)
        return self.classifier(features)
```

### NumPy数组处理
```python
import numpy as np

# 处理numpy格式的深度图
depth_maps = np.random.randn(10, 128, 128)  # [N, H, W]

encoder = SemanticEncoder(pretrained_path="encoder_weights.pth")
embeddings = encoder.encode_numpy(
    depth_maps,
    batch_size=5,
    normalize=True  # 自动归一化到[-1,1]
)
print(f"numpy输入: {depth_maps.shape} -> 输出: {embeddings.shape}")
```

## ✅ 验证安装
```bash
python test_semantic_encoder.py
```

应该看到：
```
✅ 所有测试通过！语义编码器功能正常
```

## 📚 详细文档
- `README_semantic_encoder.md` - 完整文档
- `usage_example.py` - 详细示例代码

## 🚨 常见问题
1. **导入错误** - 确保在正确目录下运行
2. **CUDA内存不足** - 使用 `device='cpu'` 或减小batch_size
3. **权重加载失败** - 检查权重文件路径和格式

## 💡 提示
- 训练好的AutoEncoder权重通常在 `outputs/autoencoder_*/best_model.pth`
- 提取的encoder权重会自动保存为 `encoder_weights_*.pth`
- 所有函数都自动处理设备管理（CPU/GPU） 