import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Optional, Tuple, List
import cv2
from PIL import Image

from modeldouble import AutoEncoder
from dataset import DepthImageDataset, create_dataloader


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 加载模型
        self.model, self.config = self._load_model(model_path)
        self.model.eval()
        
        print(f"模型加载成功，设备: {self.device}")
        print(f"潜在维度: {self.config.get('latent_dim', 768)}")
    
    def _load_model(self, model_path: str) -> Tuple[AutoEncoder, dict]:
        """加载训练好的模型"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取配置
        config = checkpoint.get('config', {})
        
        # 创建模型
        model = AutoEncoder(
            input_channels=config.get('input_channels', 1),
            latent_dim=config.get('latent_dim', 768),
            output_size=config.get('image_size', 128)
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果直接保存的是权重
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model, config
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码单张图像到潜在空间"""
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)  # 添加batch维度
            
            image = image.to(self.device)
            latent = self.model.encode(image)
            return latent
    
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """从潜在向量解码图像"""
        with torch.no_grad():
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)  # 添加batch维度
            
            latent = latent.to(self.device)
            reconstructed = self.model.decode(latent)
            return reconstructed
    
    def reconstruct_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """重建图像"""
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            reconstructed, latent = self.model(image)
            return reconstructed, latent
    
    def evaluate_dataset(self, dataloader) -> dict:
        """评估整个数据集"""
        self.model.eval()
        
        total_mse = 0
        total_l1 = 0
        total_samples = 0
        
        all_latents = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                data = data.to(self.device)
                reconstructed, latent = self.model(data)
                
                # 计算损失（只在有效区域）
                valid_mask = (data > -1).float()
                mse = nn.functional.mse_loss(reconstructed * valid_mask, data * valid_mask)
                l1 = nn.functional.l1_loss(reconstructed * valid_mask, data * valid_mask)
                
                batch_size = data.size(0)
                total_mse += mse.item() * batch_size
                total_l1 += l1.item() * batch_size
                total_samples += batch_size
                
                # 收集潜在向量
                all_latents.append(latent.cpu())
                
                if batch_idx % 10 == 0:
                    print(f"评估进度: {batch_idx}/{len(dataloader)}")
        
        # 计算平均指标
        avg_mse = total_mse / total_samples
        avg_l1 = total_l1 / total_samples
        avg_psnr = 20 * np.log10(2.0) - 10 * np.log10(avg_mse)  # 假设数据范围是[-1,1]
        
        # 合并所有潜在向量
        all_latents = torch.cat(all_latents, dim=0)
        
        results = {
            'mse': avg_mse,
            'l1': avg_l1,
            'psnr': avg_psnr,
            'num_samples': total_samples,
            'latent_stats': {
                'mean': all_latents.mean().item(),
                'std': all_latents.std().item(),
                'min': all_latents.min().item(),
                'max': all_latents.max().item()
            }
        }
        
        return results
    
    def visualize_reconstruction(self, dataloader, num_samples: int = 8, save_path: str = 'reconstruction_results.png'):
        """可视化重建结果"""
        self.model.eval()
        
        # 获取样本
        data_iter = iter(dataloader)
        data = next(data_iter)[:num_samples]
        
        with torch.no_grad():
            data = data.to(self.device)
            reconstructed, latent = self.model(data)
        
        # 移动到CPU进行可视化
        data = data.cpu()
        reconstructed = reconstructed.cpu()
        
        # 创建可视化
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
        
        for i in range(num_samples):
            # 原图
            orig_img = data[i, 0].numpy()
            axes[0, i].imshow(orig_img, cmap='viridis', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # 重建图
            recon_img = reconstructed[i, 0].numpy()
            axes[1, i].imshow(recon_img, cmap='viridis', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
            
            # 差异图
            diff_img = np.abs(orig_img - recon_img)
            valid_mask = orig_img > -1
            diff_img[~valid_mask] = 0
            axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=0.5)
            axes[2, i].set_title(f'Difference {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"重建结果已保存到: {save_path}")
    
    def analyze_latent_space(self, dataloader, save_path: str = 'latent_analysis.png'):
        """分析潜在空间"""
        self.model.eval()
        
        all_latents = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                _, latent = self.model(data)
                all_latents.append(latent.cpu())
        
        all_latents = torch.cat(all_latents, dim=0).numpy()
        
        # 分析潜在空间统计
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 潜在向量的分布直方图
        axes[0, 0].hist(all_latents.flatten(), bins=100, alpha=0.7)
        axes[0, 0].set_title('Latent Values Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # 每个维度的平均值
        dim_means = np.mean(all_latents, axis=0)
        axes[0, 1].plot(dim_means)
        axes[0, 1].set_title('Mean Values per Dimension')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Mean Value')
        
        # 每个维度的标准差
        dim_stds = np.std(all_latents, axis=0)
        axes[1, 0].plot(dim_stds)
        axes[1, 0].set_title('Standard Deviation per Dimension')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Std Dev')
        
        # 协方差矩阵的可视化（采样部分维度）
        sample_dims = min(50, all_latents.shape[1])
        sampled_latents = all_latents[:, :sample_dims]
        cov_matrix = np.cov(sampled_latents.T)
        im = axes[1, 1].imshow(cov_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title(f'Covariance Matrix (first {sample_dims} dims)')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"潜在空间分析结果已保存到: {save_path}")
        
        # 打印统计信息
        print(f"\n潜在空间统计:")
        print(f"形状: {all_latents.shape}")
        print(f"平均值: {np.mean(all_latents):.4f}")
        print(f"标准差: {np.std(all_latents):.4f}")
        print(f"最小值: {np.min(all_latents):.4f}")
        print(f"最大值: {np.max(all_latents):.4f}")
    
    def interpolate_between_images(self, img1: torch.Tensor, img2: torch.Tensor, 
                                 num_steps: int = 8, save_path: str = 'interpolation.png'):
        """在两张图像的潜在空间中进行插值"""
        self.model.eval()
        
        with torch.no_grad():
            # 编码两张图像
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)
            
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            latent1 = self.model.encode(img1)
            latent2 = self.model.encode(img2)
            
            # 插值
            alphas = np.linspace(0, 1, num_steps)
            interpolated_images = []
            
            for alpha in alphas:
                interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
                interpolated_img = self.model.decode(interpolated_latent)
                interpolated_images.append(interpolated_img.cpu())
            
            # 可视化
            fig, axes = plt.subplots(2, num_steps, figsize=(num_steps * 2, 4))
            
            # 第一行：原图和插值结果
            axes[0, 0].imshow(img1.cpu()[0, 0].numpy(), cmap='viridis', vmin=-1, vmax=1)
            axes[0, 0].set_title('Image 1')
            axes[0, 0].axis('off')
            
            for i, img in enumerate(interpolated_images[1:-1], 1):
                axes[0, i].imshow(img[0, 0].numpy(), cmap='viridis', vmin=-1, vmax=1)
                axes[0, i].set_title(f'α={alphas[i]:.2f}')
                axes[0, i].axis('off')
            
            axes[0, -1].imshow(img2.cpu()[0, 0].numpy(), cmap='viridis', vmin=-1, vmax=1)
            axes[0, -1].set_title('Image 2')
            axes[0, -1].axis('off')
            
            # 第二行：潜在空间距离
            for i, img in enumerate(interpolated_images):
                axes[1, i].imshow(img[0, 0].numpy(), cmap='viridis', vmin=-1, vmax=1)
                axes[1, i].set_title(f'Step {i}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"插值结果已保存到: {save_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AutoEncoder模型评估脚本')
    
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--data_dir', type=str, required=True, help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='结果输出目录')
    parser.add_argument('--device', type=str, help='计算设备 (cuda/cpu)')
    
    # 评估选项
    parser.add_argument('--eval_dataset', action='store_true', help='评估整个数据集')
    parser.add_argument('--visualize', action='store_true', help='可视化重建结果')
    parser.add_argument('--analyze_latent', action='store_true', help='分析潜在空间')
    parser.add_argument('--interpolate', action='store_true', help='进行图像插值')
    parser.add_argument('--num_vis_samples', type=int, default=8, help='可视化样本数量')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, args.device)
    
    # 创建数据加载器
    print("加载测试数据...")
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,  # 测试时不打乱
        num_workers=args.num_workers,
        augment=False   # 测试时不使用数据增广
    )
    
    print(f"测试数据样本数: {len(dataloader.dataset)}")
    
    # 执行评估任务
    if args.eval_dataset:
        print("\n评估数据集...")
        results = evaluator.evaluate_dataset(dataloader)
        
        print(f"\n评估结果:")
        print(f"MSE: {results['mse']:.6f}")
        print(f"L1: {results['l1']:.6f}")
        print(f"PSNR: {results['psnr']:.2f} dB")
        print(f"样本数: {results['num_samples']}")
        print(f"潜在向量统计:")
        for key, value in results['latent_stats'].items():
            print(f"  {key}: {value:.4f}")
        
        # 保存结果
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    if args.visualize:
        print("\n生成可视化结果...")
        evaluator.visualize_reconstruction(
            dataloader, 
            num_samples=args.num_vis_samples,
            save_path=str(output_dir / 'reconstruction_results.png')
        )
    
    if args.analyze_latent:
        print("\n分析潜在空间...")
        evaluator.analyze_latent_space(
            dataloader,
            save_path=str(output_dir / 'latent_analysis.png')
        )
    
    if args.interpolate:
        print("\n执行图像插值...")
        # 获取两张样本图像进行插值
        data_iter = iter(dataloader)
        batch = next(data_iter)
        
        if len(batch) >= 2:
            evaluator.interpolate_between_images(
                batch[0], batch[1],
                num_steps=8,
                save_path=str(output_dir / 'interpolation.png')
            )
        else:
            print("需要至少2张图像进行插值")
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    main() 