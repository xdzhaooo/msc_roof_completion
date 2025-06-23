# AutoEncoder for Depth Images

一个基于PyTorch的现代AutoEncoder实现，专门用于处理uint16类型的128x128深度图像，具有768维潜在空间编码。

## 功能特点

- **现代架构**: 使用ResNet风格的残差块和自注意力机制
- **深度图像处理**: 专门针对uint16类型的单通道深度图像优化
- **数据增广**: 支持90度旋转数据增广
- **高级训练技巧**: 包含梯度裁剪、学习率调度、早停机制等
- **继续训练**: 支持从检查点继续训练
- **可视化分析**: 提供详细的训练监控和结果分析工具

## 环境要求

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── model.py          # AutoEncoder模型定义
├── dataset.py        # 数据集类和数据加载器
├── train.py          # 训练脚本
├── evaluate.py       # 评估和推理脚本
├── autoencoder.py    # 原始的归一化类
├── requirements.txt  # 依赖包列表
└── README.md         # 使用说明
```

## 模型架构

### Encoder
- 基于ResNet的残差块结构
- 4个下采样阶段 (64→128→256→512 channels)
- 自注意力机制增强特征表示
- 全局平均池化 + 全连接层映射到768维潜在空间

### Decoder
- 转置卷积进行上采样重建
- 4个上采样阶段恢复到128x128分辨率
- Tanh激活函数确保输出范围[-1,1]

## 使用方法

### 1. 训练模型

基本训练命令：
```bash
python train.py --data_dir /path/to/your/data --batch_size 32 --num_epochs 100
```

完整的训练参数：
```bash
python train.py \
    --data_dir /path/to/your/data \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --latent_dim 768 \
    --optimizer AdamW \
    --scheduler CosineAnnealingLR \
    --gradient_clip 1.0 \
    --patience 20 \
    --augment \
    --output_dir ./outputs
```

### 2. 继续训练

从检查点继续训练：
```bash
python train.py \
    --data_dir /path/to/your/data \
    --resume ./outputs/autoencoder_20231201_123456/best_model.pth \
    --batch_size 32
```

### 3. 模型评估

评估训练好的模型：
```bash
python evaluate.py \
    --model_path ./outputs/autoencoder_20231201_123456/best_model.pth \
    --data_dir /path/to/test/data \
    --eval_dataset \
    --visualize \
    --analyze_latent \
    --interpolate \
    --output_dir ./evaluation_results
```

### 4. 数据格式

数据要求：
- 格式：支持PNG, TIFF, JPG等常见图像格式
- 类型：uint16单通道灰度图像
- 尺寸：任意尺寸（会自动resize到128x128）
- 组织：所有图像文件放在一个文件夹下

数据目录结构示例：
```
data/
├── image_001.png
├── image_002.png
├── image_003.tiff
└── ...
```

## 主要参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必需 | 训练数据目录 |
| `--batch_size` | 32 | 批次大小 |
| `--num_epochs` | 100 | 训练轮数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--latent_dim` | 768 | 潜在向量维度 |
| `--optimizer` | AdamW | 优化器(AdamW/Adam/SGD) |
| `--scheduler` | CosineAnnealingLR | 学习率调度器 |
| `--gradient_clip` | 1.0 | 梯度裁剪阈值 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--patience` | 20 | 早停耐心值 |
| `--augment` | True | 使用数据增广 |
| `--use_perceptual_loss` | False | 使用感知损失 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必需 | 模型权重文件路径 |
| `--data_dir` | 必需 | 测试数据目录 |
| `--eval_dataset` | False | 评估整个数据集 |
| `--visualize` | False | 可视化重建结果 |
| `--analyze_latent` | False | 分析潜在空间 |
| `--interpolate` | False | 图像插值实验 |

## 输出文件说明

### 训练输出

训练过程会在输出目录创建以下文件：
```
outputs/autoencoder_20231201_123456/
├── config.json              # 训练配置
├── best_model.pth           # 最佳模型(完整检查点)
├── best_model_weights.pth   # 最佳模型权重
├── latest_checkpoint.pth    # 最新检查点
├── checkpoint_epoch_*.pth   # 定期保存的检查点
└── tensorboard/             # TensorBoard日志
```

### 评估输出

评估会生成以下分析结果：
```
evaluation_results/
├── evaluation_results.json    # 定量评估结果
├── reconstruction_results.png # 重建效果可视化
├── latent_analysis.png       # 潜在空间分析
└── interpolation.png         # 图像插值结果
```

## 监控训练过程

使用TensorBoard监控训练：
```bash
tensorboard --logdir ./outputs/autoencoder_20231201_123456/tensorboard
```

可以查看：
- 训练/验证损失曲线
- 学习率变化
- MSE和L1损失分解

## 模型使用示例

### 编码图像到潜在空间

```python
from evaluate import ModelEvaluator
import torch

# 加载模型
evaluator = ModelEvaluator('path/to/model.pth')

# 编码图像
image = torch.randn(1, 1, 128, 128)  # 示例图像
latent_vector = evaluator.encode_image(image)
print(f"潜在向量形状: {latent_vector.shape}")  # [1, 768]
```

### 从潜在向量重建图像

```python
# 从潜在向量重建
reconstructed = evaluator.decode_latent(latent_vector)
print(f"重建图像形状: {reconstructed.shape}")  # [1, 1, 128, 128]
```

## 性能优化建议

1. **GPU加速**: 确保使用GPU训练，显著提升训练速度
2. **批次大小**: 根据GPU内存调整batch_size，通常32-64效果较好
3. **数据加载**: 使用多进程数据加载(num_workers=4-8)
4. **混合精度**: 对于较新的GPU，可以考虑使用AMP加速训练

## 常见问题

### Q: 训练过程中内存不足怎么办？
A: 减小batch_size，或者使用梯度累积技术。

### Q: 如何调整模型的潜在维度？
A: 修改`--latent_dim`参数，注意需要重新训练模型。

### Q: 数据归一化有问题怎么办？
A: 检查数据格式是否为uint16，调整`--height_range`参数。

### Q: 如何提高重建质量？
A: 可以尝试启用感知损失(`--use_perceptual_loss`)，或者增加训练轮数。

## 更新日志

- v1.0: 初始版本，包含基本的训练和评估功能
- 支持ResNet+注意力机制的现代架构
- 完整的训练流程和可视化工具

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License 