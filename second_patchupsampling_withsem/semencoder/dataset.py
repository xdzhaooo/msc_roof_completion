import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import random
from typing import List, Tuple, Optional


class DepthImageDataset(Dataset):
    """
    处理uint16类型的深度图像数据集
    支持90度旋转数据增广和归一化到[-1,1]范围
    """
    
    def __init__(self, 
                 data_dir: str, 
                 image_size: int = 128,
                 augment: bool = True,
                 height_range: float = 10.0):
        """
        Args:
            data_dir: 图像文件夹路径
            image_size: 图像尺寸，默认128x128
            augment: 是否使用数据增广
            height_range: 高度归一化范围
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        self.height_range = height_range
        
        # 支持的图像格式
        self.supported_formats = ('.png', '.tiff', '.tif', '.jpg', '.jpeg')
        
        # 获取所有图像文件路径
        self.image_paths = self._get_image_paths()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在目录 {data_dir} 中没有找到支持的图像文件")
        
        print(f"找到 {len(self.image_paths)} 张图像")
        
        # 基础变换（调整大小）
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        ])
        
        # 数据增广变换
        self.augment_transforms = [
            lambda x: x,  # 原图
            lambda x: x.rotate(90),  # 90度旋转
            lambda x: x.rotate(180),  # 180度旋转
            lambda x: x.rotate(270),  # 270度旋转
        ]
    
    def _get_image_paths(self) -> List[str]:
        """获取所有支持格式的图像文件路径"""
        image_paths = []
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith(self.supported_formats):
                image_paths.append(os.path.join(self.data_dir, filename))
        return sorted(image_paths)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像文件 uin16 格式,single channel"""
        try:
            image = Image.open(image_path).convert('I;16')
            image_array = np.array(image, dtype=np.uint16)

            return image_array
        except Exception as e:
            raise ValueError(f"加载图像 {image_path} 失败: {e}")
    
    def _normalize_height(self, height) -> torch.Tensor:
        """归一化高度数据到[-1, 1]范围"""
        height_range = 10.0
        if torch.all(height == 0):
            self.height_range = torch.tensor(height_range, dtype=torch.float32)
            self.mid_height = torch.tensor(0, dtype=torch.float32)
            self.valid_mask = height
            return height
        
        height_original = height.clone()

        mask = height > 0
        max_h, min_h = torch.max(height[mask]), torch.min(height[mask])
        mid_h = (max_h + min_h) / 2
        diff_h = max_h - min_h
        height -= mid_h
        if diff_h > height_range:
            height_range = diff_h + 1e-8
        height /= height_range
        height *= 2  # Scale to [-1, 1] range
        height[~mask] = -1

        self.height_range = torch.tensor(height_range, dtype=torch.float32)
        self.mid_height = mid_h
        self.valid_mask = mask

        if torch.all(height == -1):
            import torchvision.utils as save_image
            save_image(height_original, "height_original.png")

        return height
    
    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """应用数据增广"""
        if self.augment and len(self.augment_transforms) > 0:
            # 随机选择一个变换
            transform = random.choice(self.augment_transforms)
            image = transform(image)
        return image
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """获取单个样本"""
        image_path = self.image_paths[idx]
        
        # 加载图像
        image_array = self._load_image(image_path)
        
        # 转换为PIL图像进行变换
        image = Image.fromarray(image_array)
        
        # 应用基础变换（调整大小）
        image = self.base_transform(image)
        
        # 应用数据增广
        image = self._apply_augmentation(image)


        image_array = np.array(image)/256
        image_tensor = torch.tensor(image_array, dtype=torch.float32)


        
        # 归一化到[-1, 1]
        normalized_image = self._normalize_height(image_tensor)
        
        # 转换为torch tensor并添加通道维度
        tensor = torch.tensor(normalized_image, dtype=torch.float32).unsqueeze(0)
        
        return tensor
    
    def get_sample_image(self, idx: int = 0) -> Tuple[np.ndarray, torch.Tensor]:
        """获取样本图像用于可视化"""
        if idx >= len(self.image_paths):
            idx = 0
        
        image_path = self.image_paths[idx]
        original_image = self._load_image(image_path)
        processed_tensor = self[idx]
        
        return original_image, processed_tensor


def create_dataloader(data_dir: str,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     image_size: int = 128,
                     augment: bool = True,
                     height_range: float = 10.0) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
        image_size: 图像尺寸
        augment: 是否使用数据增广
        height_range: 高度归一化范围
    
    Returns:
        DataLoader对象
    """
    dataset = DepthImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        augment=augment,
        height_range=height_range
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if shuffle else False
    )
    
    return dataloader


def visualize_samples(dataloader: DataLoader, num_samples: int = 4):
    """可视化数据集样本"""
    import matplotlib.pyplot as plt
    
    # 获取一个批次的数据
    batch = next(iter(dataloader))
    
    # 可视化前几个样本
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
    
    for i in range(min(num_samples, batch.size(0))):
        # 原始数据（归一化后）
        img = batch[i, 0].cpu().numpy()
        
        # 显示归一化后的图像
        axes[0, i].imshow(img, cmap='viridis', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Normalized Sample {i+1}')
        axes[0, i].axis('off')
        
        # 显示有效区域（非-1的像素）
        valid_mask = img > -1
        axes[1, i].imshow(valid_mask, cmap='gray')
        axes[1, i].set_title(f'Valid Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("样本可视化已保存为 'dataset_samples.png'")


if __name__ == "__main__":
    # 测试数据集
    data_dir = "path/to/your/data"  # 请替换为实际的数据路径
    
    try:
        # 创建数据加载器
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # 在测试时使用0避免多进程问题
            augment=True
        )
        
        print(f"数据集大小: {len(dataloader.dataset)}")
        print(f"批次数量: {len(dataloader)}")
        
        # 测试加载一个批次
        batch = next(iter(dataloader))
        print(f"批次形状: {batch.shape}")
        print(f"数据类型: {batch.dtype}")
        print(f"数值范围: [{batch.min():.3f}, {batch.max():.3f}]")
        
        # 可视化样本（如果安装了matplotlib）
        try:
            visualize_samples(dataloader)
        except ImportError:
            print("未安装matplotlib，跳过可视化")
            
    except FileNotFoundError:
        print(f"数据目录 {data_dir} 不存在，请检查路径") 