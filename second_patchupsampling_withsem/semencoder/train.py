import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from model import AutoEncoder, count_parameters
from dataset import create_dataloader, visualize_samples


class PerceptualLoss(nn.Module):
    """感知损失，使用预训练的特征提取器"""
    def __init__(self, feature_layers=[2, 7, 12, 21, 30]):
        super(PerceptualLoss, self).__init__()
        from torchvision.models import vgg19
        vgg = vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        
        # 提取特定层
        self.features = nn.ModuleList()
        for i, layer in enumerate(vgg):
            self.features.append(layer)
            if i == max(feature_layers):
                break
        
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # 确保模型在正确的设备上
        device = pred.device
        if next(self.features.parameters()).device != device:
            self.features = self.features.to(device)
            
        # 将单通道图像复制为三通道以适配VGG
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        pred_features = []
        target_features = []
        
        x_pred, x_target = pred, target
        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # 计算特征损失
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += nn.functional.mse_loss(pred_feat, target_feat)
        
        return loss


class CombinedLoss(nn.Module):
    """组合损失函数，专门针对深度图像优化"""
    def __init__(self, use_perceptual=False, perceptual_weight=0.1, mse_weight=1.0, 
                 ssim_weight=0.3, edge_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        if use_perceptual:
            try:
                self.perceptual_loss = PerceptualLoss()
            except:
                print("无法加载感知损失，使用MSE+L1损失")
                self.use_perceptual = False
    
    def ssim_loss(self, pred, target, window_size=11):
        """计算SSIM损失"""
        def gaussian_kernel(size, sigma=1.5):
            coords = torch.arange(size).float() - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.unsqueeze(0).unsqueeze(0)
        
        # 创建高斯核
        kernel = gaussian_kernel(window_size)
        kernel = kernel.expand(pred.size(1), 1, window_size, window_size).to(pred.device)
        
        # 计算均值
        mu1 = F.conv2d(pred, kernel, padding=window_size//2, groups=pred.size(1))
        mu2 = F.conv2d(target, kernel, padding=window_size//2, groups=target.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size//2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=window_size//2, groups=target.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=window_size//2, groups=pred.size(1)) - mu1_mu2
        
        # SSIM计算
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()
    
    def edge_loss(self, pred, target):
        """计算边缘损失"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_x = sobel_x.to(pred.device)
        sobel_y = sobel_y.to(pred.device)
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.l1_loss(pred_grad, target_grad)
    
    def forward(self, pred, target, valid_mask=None):
        # 如果有有效掩码，只计算有效区域的损失
        if valid_mask is not None:
            pred_valid = pred * valid_mask
            target_valid = target * valid_mask
        else:
            pred_valid = pred
            target_valid = target
            valid_mask = torch.ones_like(pred)
        
        # 基础损失
        mse_loss = self.mse_loss(pred_valid, target_valid) * self.mse_weight
        l1_loss = self.l1_loss(pred_valid, target_valid)
        
        total_loss = mse_loss + l1_loss
        
        # SSIM损失
        if self.ssim_weight > 0:
            ssim_loss = self.ssim_loss(pred_valid, target_valid) * self.ssim_weight
            total_loss += ssim_loss
        else:
            ssim_loss = torch.tensor(0.0)
        
        # 边缘损失
        if self.edge_weight > 0:
            edge_loss = self.edge_loss(pred_valid, target_valid) * self.edge_weight
            total_loss += edge_loss
        else:
            edge_loss = torch.tensor(0.0)
        
        # 感知损失（可选）
        perceptual_loss = torch.tensor(0.0)
        if self.use_perceptual:

            perceptual_loss = self.perceptual_loss(pred, target) * self.perceptual_weight
            total_loss += perceptual_loss
            assert perceptual_loss!= torch.tensor(0.0), "感知损失计算失败，请检查PerceptualLoss类"
        
        return total_loss, mse_loss, l1_loss, ssim_loss, edge_loss, perceptual_loss


class AverageMeter:
    """记录平均值的工具类"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AutoEncoderTrainer:
    """AutoEncoder训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化模型
        self.model = AutoEncoder(
            input_channels=config['input_channels'],
            latent_dim=config['latent_dim'],
            output_size=config['image_size']
        ).to(self.device)
        
        print(f"模型参数数量: {count_parameters(self.model):,}")
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()

        #print mean grad of model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"参数: {name}, 平均梯度: {param.grad.mean() if param.grad is not None else '无梯度'}")
        
        # 初始化损失函数
        self.criterion = CombinedLoss(
            #use_perceptual=config.get('use_perceptual_loss', False),
            use_perceptual=True,
            perceptual_weight=config.get('perceptual_weight', 0.1),
            mse_weight=config.get('mse_weight', 1.0),
            #ssim_weight=config.get('ssim_weight', 0.3),
            #edge_weight=config.get('edge_weight', 0.2)
            ssim_weight=0,
            edge_weight=0
        )
        
        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 初始化tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        # 早停机制
        self.patience_counter = 0
        
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_type = self.config.get('optimizer', 'AdamW')
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'CosineAnnealingLR')
        
        if scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['num_epochs']
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5
            )
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=20, 
                gamma=0.1
            )
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.config['num_epochs']
            )
        else:
            return None
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            # 只保存模型权重
            torch.save(self.model.state_dict(), self.output_dir / 'best_model_weights.pth')
        
        # 定期保存检查点
        if epoch % self.config.get('save_frequency', 10) == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.training_history = checkpoint.get('training_history', {'train_loss': [], 'val_loss': []})
            
            print(f"成功加载检查点，从第 {self.current_epoch} 轮继续训练")
            return True
            
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        l1_meter = AverageMeter()
        ssim_meter = AverageMeter()
        edge_meter = AverageMeter()
        perceptual_meter = AverageMeter()
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)
            
            # 创建有效掩码（非-1的像素）
            valid_mask = (data >= -1).float()

            
            # 前向传播
            self.optimizer.zero_grad()
            reconstructed, latent = self.model(data)


            
            # 计算损失
            total_loss, mse_loss, l1_loss, ssim_loss, edge_loss, perceptual_loss = self.criterion(reconstructed, data, valid_mask)
            
            # 反向传播
            total_loss.backward()
            
            
            self.optimizer.step()
            
            # 更新统计
            batch_size = data.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            mse_meter.update(mse_loss.item(), batch_size)
            l1_meter.update(l1_loss.item(), batch_size)
            ssim_meter.update(ssim_loss.item(), batch_size)
            edge_meter.update(edge_loss.item(), batch_size)
            perceptual_meter.update(perceptual_loss.item(), batch_size)
            
            # 打印进度
            if batch_idx % self.config.get('log_interval', 100) == 0:
                print(f'训练 Epoch: {self.current_epoch} [{batch_idx * batch_size}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {total_loss.item():.6f}')
                print(f'MSE Loss: {mse_loss.item():.6f}, L1 Loss: {l1_loss.item():.6f}, '
                      f'SSIM Loss: {ssim_loss.item():.6f}, Edge Loss: {edge_loss.item():.6f}, '
                      f'Perceptual Loss: {perceptual_loss.item():.6f}')
                from torchvision.utils import save_image
                save_image((data+1)/2,"data.png")
                save_image((reconstructed+1)/2,"reconstructed.png")
                save_image((valid_mask),"valid_mask.png")
        
        return {
            'total_loss': loss_meter.avg,
            'mse_loss': mse_meter.avg,
            'l1_loss': l1_meter.avg,
            'ssim_loss': ssim_meter.avg,
            'edge_loss': edge_meter.avg,
            'perceptual_loss': perceptual_meter.avg
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        l1_meter = AverageMeter()
        ssim_meter = AverageMeter()
        edge_meter = AverageMeter()
        perceptual_meter = AverageMeter()
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                valid_mask = (data > -1).float()
                
                reconstructed, latent = self.model(data)
                total_loss, mse_loss, l1_loss, ssim_loss, edge_loss, perceptual_loss = self.criterion(reconstructed, data, valid_mask)
                
                batch_size = data.size(0)
                loss_meter.update(total_loss.item(), batch_size)
                mse_meter.update(mse_loss.item(), batch_size)
                l1_meter.update(l1_loss.item(), batch_size)
                ssim_meter.update(ssim_loss.item(), batch_size)
                edge_meter.update(edge_loss.item(), batch_size)
                perceptual_meter.update(perceptual_loss.item(), batch_size)
        
        return {
            'total_loss': loss_meter.avg,
            'mse_loss': mse_meter.avg,
            'l1_loss': l1_meter.avg,
            'ssim_loss': ssim_meter.avg,
            'edge_loss': edge_meter.avg,
            'perceptual_loss': perceptual_meter.avg
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """主训练循环"""
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(train_dataloader.dataset)}")
        if val_dataloader:
            print(f"验证样本数: {len(val_dataloader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_dataloader)
            
            # 验证
            val_metrics = None
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'] if val_metrics else train_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # 记录训练历史
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['total_loss'])
            
            # TensorBoard日志
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Train/Total_Loss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Train/MSE_Loss', train_metrics['mse_loss'], epoch)
            self.writer.add_scalar('Train/L1_Loss', train_metrics['l1_loss'], epoch)
            self.writer.add_scalar('Train/SSIM_Loss', train_metrics['ssim_loss'], epoch)
            self.writer.add_scalar('Train/Edge_Loss', train_metrics['edge_loss'], epoch)
            self.writer.add_scalar('Train/Perceptual_Loss', train_metrics['perceptual_loss'], epoch)
            
            if val_metrics:
                self.writer.add_scalar('Val/Total_Loss', val_metrics['total_loss'], epoch)
                self.writer.add_scalar('Val/MSE_Loss', val_metrics['mse_loss'], epoch)
                self.writer.add_scalar('Val/L1_Loss', val_metrics['l1_loss'], epoch)
                self.writer.add_scalar('Val/SSIM_Loss', val_metrics['ssim_loss'], epoch)
                self.writer.add_scalar('Val/Edge_Loss', val_metrics['edge_loss'], epoch)
                self.writer.add_scalar('Val/Perceptual_Loss', val_metrics['perceptual_loss'], epoch)
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch {epoch+1}/{self.config["num_epochs"]} - {epoch_time:.2f}s')
            print(f'训练损失: {train_metrics["total_loss"]:.6f}')
            if val_metrics:
                print(f'验证损失: {val_metrics["total_loss"]:.6f}')
            print(f'学习率: {current_lr:.6f}')
            
            # 保存模型
            current_loss = val_metrics['total_loss'] if val_metrics else train_metrics['total_loss']
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, current_loss, is_best)
            
            # 早停检查
            patience = self.config.get('patience', 0)
            if patience > 0 and self.patience_counter >= patience:
                print(f'早停: {patience} 轮验证损失未改善')
                break
            
            print('-' * 80)
        
        # 训练结束
        total_time = time.time() - start_time
        print(f'\n训练完成! 总耗时: {total_time/3600:.2f} 小时')
        print(f'最佳验证损失: {self.best_loss:.6f}')
        
        self.writer.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AutoEncoder训练脚本')
    
    # 数据相关
    parser.add_argument('--data_dir', type=str,  default='./heightmap', help='数据目录路径')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--image_size', type=int, default=128, help='图像尺寸')
    parser.add_argument('--height_range', type=float, default=10.0, help='高度归一化范围')
    
    # 模型相关
    parser.add_argument('--latent_dim', type=int, default=768, help='潜在向量维度')
    parser.add_argument('--input_channels', type=int, default=1, help='输入通道数')
    
    # 训练相关
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'], help='优化器')
    parser.add_argument('--scheduler', type=str, default='StepLR', help='学习率调度器')
    
    # 损失函数相关
    parser.add_argument('--use_perceptual_loss', action='store_true', help='使用感知损失')
    parser.add_argument('--perceptual_weight', type=float, default=0.01, help='感知损失权重')
    parser.add_argument('--mse_weight', type=float, default=1.0, help='MSE损失权重')
    parser.add_argument('--ssim_weight', type=float, default=0.3, help='SSIM损失权重')
    parser.add_argument('--edge_weight', type=float, default=0.2, help='边缘损失权重')
    
    # 其他
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--resume', type=str,default='outputs/autoencoder_20250601_134120/checkpoint_epoch_15.pth' ,help='继续训练的检查点路径')
    parser.add_argument('--num_workers', type=int, default=12, help='数据加载器工作进程数')
    parser.add_argument('--augment', action='store_true', default=True, help='使用数据增广')
    parser.add_argument('--patience', type=int, default=20, help='早停的耐心值')
    parser.add_argument('--save_frequency', type=int, default=5, help='保存频率（轮数）')
    parser.add_argument('--log_interval', type=int, default=10, help='日志打印间隔')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建配置字典
    config = vars(args)
    
    # 添加时间戳到输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = os.path.join(args.output_dir, f'autoencoder_{timestamp}')
    
    # 创建数据加载器
    print("创建数据加载器...")
    
    # 这里简化处理，实际使用时可以实现训练/验证集分割
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=args.augment,
        height_range=args.height_range
    )
    
    # 如果需要验证集，可以在这里创建
    val_dataloader = None
    # val_dataloader = create_dataloader(...) # 使用不同的数据路径或分割
    
    # 创建训练器
    trainer = AutoEncoderTrainer(config)
    
    # 如果指定了继续训练的检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(train_dataloader, val_dataloader)


if __name__ == '__main__':
    main() 