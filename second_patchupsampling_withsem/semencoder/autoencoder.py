import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

# -------------------------------
# Normalization Method for Depth
# -------------------------------
class HeightNormalize(object):
    def __init__(self):
        self.height_range = -1
        self.mid_height = -1
        self.valid_mask = None

    def __call__(self, height: torch.Tensor, height_range: float = 10):
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

# -----------------------
# Core Network Modules
# -----------------------

# 增强的细节保持模块 - 专注高频特征
class EnhancedDetailModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 高频细节提取 - 更激进的细节学习
        self.high_freq = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1, bias=False),
            nn.BatchNorm2d(channels//2),
            nn.GELU(),
        )
        
        # 使用空洞卷积捕获不同尺度的细节
        self.dilated_conv1 = nn.Conv2d(channels//2, channels//4, 3, padding=1, dilation=1, bias=False)
        self.dilated_conv2 = nn.Conv2d(channels//2, channels//4, 3, padding=2, dilation=2, bias=False)
        
        # 边缘增强
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.GELU(),
        )
        
        # 纹理增强
        self.texture_enhance = nn.Sequential(
            nn.Conv2d(channels, channels//4, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.GELU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        
    def forward(self, x):
        # 多分支细节提取
        high_base = self.high_freq(x)
        high1 = self.dilated_conv1(high_base)
        high2 = self.dilated_conv2(high_base)
        edge = self.edge_enhance(x)
        texture = self.texture_enhance(x)
        
        # 融合所有细节特征
        enhanced = torch.cat([high1, high2, edge, texture], dim=1)
        refined = self.fusion(enhanced)
        
        return x + refined  # 强残差连接保持细节

# 改进的上采样模块 - 使用转置卷积保持细节
class DetailPreservingUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # 使用转置卷积而不是插值，更好地保持细节
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=4, stride=scale_factor, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 细节精炼
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        # 转置卷积上采样
        x = F.gelu(self.bn1(self.conv_transpose(x)))
        # 细节精炼
        refined = self.refine(x)
        return F.gelu(x + refined)

# 加速收敛的网络模块改进
class FastConvergenceDepthBlock(nn.Module):
    """优化的深度特征提取模块 - 加速收敛"""
    def __init__(self, channels):
        super().__init__()
        branch_channels = channels // 4
        
        # 使用更好的初始化
        self.conv1 = nn.Conv2d(channels, branch_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, branch_channels, 5, padding=2, bias=False) 
        self.conv3 = nn.Conv2d(channels, branch_channels, 1, bias=False)
        self.conv4 = nn.Conv2d(channels, branch_channels, 3, padding=1, bias=False)
        
        self.fusion = nn.Conv2d(branch_channels * 4, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        
        # 残差缩放因子 - 帮助稳定训练
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Xavier初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        
        fused = self.fusion(torch.cat([feat1, feat2, feat3, feat4], dim=1))
        # 使用可学习的残差缩放
        return F.gelu(self.bn(fused) * self.residual_scale + x)

class LayerNormConv2d(nn.Module):
    """LayerNorm版本的2D卷积 - 通常收敛更稳定"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)  # 等效于LayerNorm
        
    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))

# -----------------------
# Fast Convergence Autoencoder
# -----------------------
class FastConvergenceAutoencoder(nn.Module):
    def __init__(self, latent_dim=768, dropout_rate=0.1):
        super().__init__()
        
        # 编码器 - 使用LayerNorm替代部分BatchNorm
        self.enc_conv1 = nn.Sequential(
            LayerNormConv2d(1, 64, 7, stride=2, padding=3),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.05)  # 轻微dropout防止过拟合
        )
        
        self.enc_conv2 = nn.Sequential(
            LayerNormConv2d(64, 128, 3, stride=2, padding=1),
            FastConvergenceDepthBlock(128),
        )
        
        self.enc_conv3 = nn.Sequential(
            LayerNormConv2d(128, 256, 3, stride=2, padding=1),
            FastConvergenceDepthBlock(256),
            EnhancedDetailModule(256),
        )
        
        self.enc_conv4 = nn.Sequential(
            LayerNormConv2d(256, 512, 3, stride=2, padding=1),
            FastConvergenceDepthBlock(512),
            EnhancedDetailModule(512),
        )
        
        # 更强的瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 768, 3, padding=1, bias=False),
            nn.GroupNorm(32, 768),  # GroupNorm通常比BatchNorm更稳定
            nn.GELU(),
            FastConvergenceDepthBlock(768),
            EnhancedDetailModule(768),
            nn.Conv2d(768, 768, 3, padding=1, bias=False),
            nn.GroupNorm(32, 768),
            nn.GELU(),
        )
        
        # 潜在空间编码器 - 更快的激活函数
        self.encoder_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(768 * 16, latent_dim * 2),
            nn.SiLU(),  # SiLU通常比GELU收敛更快
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU()
        )
        
        # 潜在空间解码器
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, 768 * 16),
            nn.SiLU()
        )
        
        # 解码器
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(768, 768, 3, padding=1, bias=False),
            nn.GroupNorm(32, 768),
            nn.GELU(),
            FastConvergenceDepthBlock(768),
            EnhancedDetailModule(768),
        )
        
        # 上采样模块
        self.dec_up4 = DetailPreservingUpsample(768, 512, scale_factor=2)
        self.dec_detail4 = EnhancedDetailModule(512)
        
        self.dec_up3 = DetailPreservingUpsample(512, 256, scale_factor=2)
        self.dec_detail3 = EnhancedDetailModule(256)
        
        self.dec_up2 = DetailPreservingUpsample(256, 128, scale_factor=2)
        self.dec_detail2 = EnhancedDetailModule(128)
        
        self.dec_up1 = DetailPreservingUpsample(128, 64, scale_factor=2)
        self.dec_detail1 = EnhancedDetailModule(64)
        
        # 最终输出层
        self.final_up = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.final_bn = nn.BatchNorm2d(32)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            EnhancedDetailModule(32),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.Conv2d(8, 1, 1),
            nn.Tanh()
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 编码器
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)
        x4 = self.enc_conv4(x3)
        
        # 瓶颈处理
        x = self.bottleneck(x4)
        
        # 潜在空间
        z = self.encoder_final(x)
        
        # 解码器
        x = self.decoder_linear(z)
        x = x.view(batch_size, 768, 4, 4)
        x = self.dec_conv1(x)
        
        # 逐步上采样
        x = self.dec_up4(x)
        x = self.dec_detail4(x)
        
        x = self.dec_up3(x)
        x = self.dec_detail3(x)
        
        x = self.dec_up2(x)
        x = self.dec_detail2(x)
        
        x = self.dec_up1(x)
        x = self.dec_detail1(x)
        
        x = F.gelu(self.final_bn(self.final_up(x)))
        x = self.final_conv(x)
        
        return z, x

    def encode(self, x):
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)
        x = self.bottleneck(x)
        return self.encoder_final(x)
    
    def decode(self, z):
        batch_size = z.size(0)
        x = self.decoder_linear(z)
        x = x.view(batch_size, 768, 4, 4)
        x = self.dec_conv1(x)
        
        x = self.dec_up4(x)
        x = self.dec_detail4(x)
        x = self.dec_up3(x)
        x = self.dec_detail3(x)
        x = self.dec_up2(x)
        x = self.dec_detail2(x)
        x = self.dec_up1(x)
        x = self.dec_detail1(x)
        
        x = F.gelu(self.final_bn(self.final_up(x)))
        return self.final_conv(x)

# -----------------------
# Loss Functions
# -----------------------

# 保留核心的梯度损失函数
class SimplifiedGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 简化的梯度损失 - 只关注主要边缘
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

class DetailReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 多尺度细节检测
        self.detail_kernels = nn.ParameterList([
            nn.Parameter(torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                        dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False),
            nn.Parameter(torch.tensor([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], 
                        dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False),
        ])
        
    def forward(self, pred, target):
        detail_loss = 0
        for kernel in self.detail_kernels:
            pred_detail = F.conv2d(pred, kernel, padding=1)
            target_detail = F.conv2d(target, kernel, padding=1)
            detail_loss += F.l1_loss(pred_detail, target_detail)
        return detail_loss / len(self.detail_kernels)

# SSIM Loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.device == img1.device:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(window_size, size_average)
        
    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)

# -----------------------
# Dataset
# -----------------------
class DepthDataset(Dataset):
    def __init__(self, image_dir, transform=None, rotate=False, target_size=(128, 128)):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.tif')]
        self.transform = transform
        self.rotate = rotate
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path)
        
        # Resize to target size using nearest neighbor interpolation
        img = img.resize(self.target_size, Image.NEAREST)
        
        # Convert to numpy array and normalize
        img = np.array(img).astype(np.float32) / 256.0  # uint16 -> float
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)

        if self.rotate:
            angle = float(np.random.choice([0, 90, 180, 270]))  # Convert to float
            img = transforms.functional.rotate(img, angle)

        img = HeightNormalize()(img)

        return img

# -----------------------
# Utility Functions
# -----------------------
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载checkpoint并恢复模型、优化器和调度器状态"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# -----------------------
# Fast Convergence Training
# -----------------------
def train_fast_convergence_autoencoder(save_dir='checkpoints_fast', save_interval=5):
    """
    快速收敛版训练：
    - 学习率预热 + 更激进的调度
    - 混合精度训练
    - 梯度累积
    - 更优的优化器配置
    - 使用FastConvergenceAutoencoder模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 增大batch size或使用梯度累积
    dataset = DepthDataset(r"heightmap", rotate=True)
    dataloader = DataLoader(dataset, batch_size=96, shuffle=True, num_workers=6, pin_memory=True)

    # 使用快速收敛优化的模型
    model = FastConvergenceAutoencoder(latent_dim=768).cuda()

    # 备选：更激进的AdamW配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3,  # 提高基础学习率
        weight_decay=1e-4,
        betas=(0.9, 0.999),  # 更激进的动量
        eps=1e-8
    )
    print("使用激进AdamW配置")
    
    # 学习率预热 + OneCycleLR调度
    total_steps = len(dataloader) * 60  # 减少到60个epoch
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,  # 降低峰值学习率
        total_steps=total_steps,
        pct_start=0.15,  # 增加预热时间
        anneal_strategy='cos',
        cycle_momentum=False,  # 关闭动量循环，更稳定
    )
    
    # 谨慎使用混合精度训练
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2.**8,  # 降低初始缩放
        growth_factor=1.5,  # 更保守的增长
        backoff_factor=0.8,
        growth_interval=1000
    )
    
    # 损失函数 - 使用更快收敛的组合
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()  # MSE通常前期收敛更快
    ssim_criterion = SSIMLoss().cuda()
    gradient_criterion = SimplifiedGradientLoss().cuda()
    detail_criterion = DetailReconstructionLoss().cuda()
    
    # 早停机制
    best_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    # 梯度累积步数
    accumulation_steps = 2
    
    print("🚀 开始快速收敛训练...")
    print(f"特性: FastConvergenceAutoencoder, AdamW优化器, OneCycleLR, 混合精度, 梯度累积")
    print(f"Batch Size: 96, Max LR: 1e-4, Epochs: 60, 早停patience: 8")

    for epoch in range(60):  # 减少训练轮数
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, img in enumerate(dataloader):
            img = img.cuda(non_blocking=True)
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                z, recon = model(img)
                
                # 计算损失 - 动态调整策略
                l1_loss = l1_criterion(recon, img)
                mse_loss = mse_criterion(recon, img)
                ssim_loss = ssim_criterion(recon, img)
                gradient_loss = gradient_criterion(recon, img)
                detail_loss = detail_criterion(recon, img)
                
                # 动态调整权重 - 早期更注重快速收敛，后期注重细节
                progress = (epoch * len(dataloader) + i) / total_steps
                
                if progress < 0.3:  # 前30%：快速收敛
                    loss = (0.6 * mse_loss +       # MSE收敛快
                           0.3 * l1_loss +        # L1保持稳定
                           0.1 * ssim_loss)       # 少量结构损失
                elif progress < 0.7:  # 中期70%：平衡优化
                    loss = (0.4 * l1_loss + 
                           0.3 * ssim_loss + 
                           0.2 * gradient_loss +
                           0.1 * detail_loss)
                else:  # 后期30%：细节优化
                    loss = (0.3 * l1_loss + 
                           0.25 * ssim_loss + 
                           0.25 * gradient_loss +
                           0.2 * detail_loss)
                
                # 梯度累积
                loss = loss / accumulation_steps

            # 混合精度反向传播
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪 - 更激进
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)  # 更小的裁剪值
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps

            if i % 15 == 0:  # 更频繁的日志
                current_lr = optimizer.param_groups[0]['lr']
                phase = "快速收敛" if progress < 0.3 else "平衡优化" if progress < 0.7 else "细节优化"
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()*accumulation_steps:.4f}, LR: {current_lr:.6f}")
                print(f"  阶段: {phase}, Progress: {progress:.2f}")
                print(f"  L1: {l1_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}")
                
                if i % 45 == 0:
                    from torchvision.utils import save_image
                    save_image((img+1)/2, "original_fast.png")
                    save_image((recon+1)/2, "reconstructed_fast.png")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'latent_dim': 768,
            }
            torch.save(best_checkpoint, os.path.join(save_dir, 'best_fast_model.pt'))
            print(f"  💾 保存最佳模型，Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏰ 早停计数: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"🛑 早停触发！最佳Loss: {best_loss:.4f}")
                break
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'latent_dim': 768,
            }
            checkpoint_path = os.path.join(save_dir, f'fast_checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"快速收敛checkpoint保存: {checkpoint_path}")

    print(f"🎉 训练完成！最佳Loss: {best_loss:.4f}")
    return best_loss

if __name__ == "__main__":
    # 快速收敛版训练
    print("🚀 使用快速收敛版训练 (768维潜在空间, 混合精度, 梯度累积)")
    train_fast_convergence_autoencoder()
        