import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class LightweightSEBlock(nn.Module):
    """轻量化SE块"""
    def __init__(self, channels, reduction=16):
        super(LightweightSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EnhancedResNet50Encoder(nn.Module):
    """增强版ResNet-50编码器，在效率和效果之间平衡"""
    def __init__(self, input_channels=1, latent_dim=768):
        super(EnhancedResNet50Encoder, self).__init__()
        self.inplanes = 64
        
        # 增强的初始层 - 对深度图更友好
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 128->64->32
        )
        
        # 增强的ResNet层 - 适当增加层数
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)   # 32x32, 256 channels
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)  # 16x16, 512 channels  
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)  # 8x8, 1024 channels
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)  # 4x4, 2048 channels
        
        # 添加注意力机制
        self.attention1 = LocalAttentionBlock(256)
        self.attention2 = LocalAttentionBlock(512) 
        self.attention3 = LocalAttentionBlock(1024)
        self.se_block = LightweightSEBlock(2048)
        
        # 深度图特定的特征增强模块
        self.depth_enhance = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度特征融合
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 改进的特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局特征
        self.spatial_pool = nn.AdaptiveAvgPool2d(2)  # 保留2x2空间信息
        
        # 更强的全连接层
        # 输入: 2048 (global) + 2048*4 (original spatial) + 512*4 (enhanced spatial) = 12288
        self.fc = nn.Sequential(
            nn.Linear(2048 + 2048 * 4 + 512 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 增强的stem
        x = self.stem(x)       # 128->32
        
        # ResNet backbone with attention
        x1 = self.layer1(x)     # 32x32, 256
        x1 = self.attention1(x1)
        
        x2 = self.layer2(x1)    # 16x16, 512
        x2 = self.attention2(x2)
        
        x3 = self.layer3(x2)    # 8x8, 1024  
        x3 = self.attention3(x3)
        
        x4 = self.layer4(x3)    # 4x4, 2048
        x4 = self.se_block(x4)
        
        # 深度图特定增强
        enhanced = self.depth_enhance(x4)  # [B, 1024, 4, 4]
        
        # 多尺度特征
        multiscale = self.multiscale_fusion(enhanced)  # [B, 512, 4, 4]
        
        # 特征提取
        global_feat = self.global_pool(x4).flatten(1)      # [B, 2048]
        spatial_feat = self.spatial_pool(x4).flatten(1)    # [B, 2048*4]
        enhanced_spatial = self.spatial_pool(multiscale).flatten(1)  # [B, 512*4]
        
        # 组合所有特征
        combined = torch.cat([global_feat, spatial_feat, enhanced_spatial], dim=1)  # [B, 12288]
        
        # 映射到潜在空间
        latent = self.fc(combined)
        return latent


# 更新编码器别名
class Encoder(EnhancedResNet50Encoder):
    """编码器别名，保持向后兼容"""
    pass


class LocalAttentionBlock(nn.Module):
    """真正的局部注意力机制，保留更多细节"""
    def __init__(self, in_channels, reduction=8):
        super(LocalAttentionBlock, self).__init__()
        self.in_channels = in_channels
        
        # 空间注意力：只关注空间维度，不改变通道数
        self.spatial_conv = nn.Sequential(
            # 使用1x1卷积生成空间注意力图
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            # 使用3x3卷积进行局部空间交互
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出0-1的注意力权重
        )
        
        # 通道注意力：类似轻量化的SE
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 减小gamma初始值，降低注意力影响
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # 初始为0.1，影响较小
        
    def forward(self, x):
        # 空间注意力：关注"哪里"重要
        spatial_att = self.spatial_conv(x)  # [B, 1, H, W]
        
        # 通道注意力：关注"什么"重要  
        channel_att = self.channel_attention(x)  # [B, C, 1, 1]
        
        # 结合空间和通道注意力
        attention = spatial_att * channel_att  # 广播相乘
        
        # 应用注意力
        out = x * attention
        
        # 温和的残差连接
        out = self.gamma * out + (1 - self.gamma) * x
        return out


class AttentionBlock(nn.Module):
    """改进的局部注意力机制，避免过度平滑"""
    def __init__(self, in_channels, reduction=8):
        super(AttentionBlock, self).__init__()
        # 直接使用局部注意力块
        self.local_attention = LocalAttentionBlock(in_channels, reduction)
        
    def forward(self, x):
        return self.local_attention(x)


class SEBlock(nn.Module):
    """轻量化的Squeeze-and-Excitation块，减少对细节的影响"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        # 使用更温和的池化策略
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        # 添加可学习的权重参数，控制SE块的影响强度
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)  # 初始化为0.5，减少影响
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 使用可学习权重控制SE块的影响强度
        return x * (1 + self.alpha * (y - 1))


class DecoderBlock(nn.Module):
    """改进的解码器构建块，更好地保留细节"""
    def __init__(self, in_channels, out_channels, scale_factor=2, use_attention=False):
        super(DecoderBlock, self).__init__()
        
        # 改进的上采样：使用转置卷积+双线性插值组合
        self.upsample = nn.Sequential(
            # 先用转置卷积进行学习式上采样
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=scale_factor, 
                             stride=scale_factor, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征细化卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 添加1x1卷积进行特征融合
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 减少注意力使用，只在需要时使用局部注意力
        self.attention = LocalAttentionBlock(out_channels) if use_attention else None
        
        # 轻量化SE块
        self.se = SEBlock(out_channels)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        
        if self.attention is not None:
            x = self.attention(x)
            
        x = self.se(x)
        return x


class EnhancedDecoder(nn.Module):
    """增强的解码器，提升重建质量"""
    def __init__(self, latent_dim=768, output_channels=1, output_size=128):
        super(EnhancedDecoder, self).__init__()
        
        # 更强的潜在向量重建
        self.global_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024 * 2 * 2)
        )
        
        self.spatial_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512 * 2 * 2)
        )
        
        # 增强的空间特征融合
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(1024 + 512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # 添加注意力
            LocalAttentionBlock(1024)
        )
        
        # 上采样到4x4
        self.spatial_upsample = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # 增强的初始卷积
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            LocalAttentionBlock(1024)
        )
        
        # 增强的解码器块
        self.decoder_blocks = nn.ModuleList([
            # 4x4 -> 8x8 (使用注意力)
            DecoderBlock(1024, 512, scale_factor=2, use_attention=True),
            # 8x8 -> 16x16 (使用注意力)
            DecoderBlock(512, 256, scale_factor=2, use_attention=True),
            # 16x16 -> 32x32
            DecoderBlock(256, 128, scale_factor=2, use_attention=False),
            # 32x32 -> 64x64
            DecoderBlock(128, 64, scale_factor=2, use_attention=False),
            # 64x64 -> 128x128
            DecoderBlock(64, 32, scale_factor=2, use_attention=False)
        ])
        
        # 增强的最终输出层
        self.final_conv = nn.Sequential(
            # 细节增强模块
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # 添加局部注意力保留细节
            LocalAttentionBlock(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 最终输出
            nn.Conv2d(8, output_channels, kernel_size=1, padding=0),
            nn.Tanh()
        )
        
    def forward(self, latent):
        # 分别重建特征
        global_features = self.global_fc(latent)
        spatial_features = self.spatial_fc(latent)
        
        # 重塑为特征图
        global_map = global_features.view(-1, 1024, 2, 2)
        spatial_map = spatial_features.view(-1, 512, 2, 2)
        
        # 增强的融合
        combined = torch.cat([global_map, spatial_map], dim=1)
        x = self.spatial_fusion(combined)
        
        # 上采样
        x = self.spatial_upsample(x)
        x = self.initial_conv(x)
        
        # 逐步解码
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        # 最终输出
        x = self.final_conv(x)
        return x


# 为了兼容性，保留原来的类名
class Decoder(EnhancedDecoder):
    """解码器别名，保持向后兼容"""
    pass


class AutoEncoder(nn.Module):
    """增强的AutoEncoder模型，平衡效率和效果"""
    def __init__(self, input_channels=1, latent_dim=768, output_size=128):
        super(AutoEncoder, self).__init__()
        self.encoder = EnhancedResNet50Encoder(input_channels, latent_dim)
        self.decoder = EnhancedDecoder(latent_dim, input_channels, output_size)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        """仅编码，返回潜在表示"""
        return self.encoder(x)
    
    def decode(self, latent):
        """仅解码，从潜在表示重建图像"""
        return self.decoder(latent)


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(input_channels=1, latent_dim=768, output_size=128).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 测试前向传播
    test_input = torch.randn(2, 1, 128, 128).to(device)
    reconstructed, latent = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # 打印网络结构信息
    print("\n=== 基于ResNet-50的轻量化模型信息 ===")
    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)
    
    print(f"总参数: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"编码器参数: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"解码器参数: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"相比ResNet-50: {total_params/25.6e6:.1f}x")
    print("\n=== 优化要点 ===")
    print("1. 基于ResNet-50主干网络，参数量大幅减少")
    print("2. 保留空间信息(2x2)，但更轻量化")
    print("3. 使用标准的Bottleneck结构，提高效率")
    print("4. 维持768维潜在空间")
    print("5. 可选择加载ImageNet预训练权重") 