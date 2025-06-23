import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from core.base_network import BaseNetwork
from torchmetrics.image import StructuralSimilarityIndexMeasure

from torchvision.models import vgg16
from torchvision.utils import save_image
from torch import nn

class VGGPerceptualLoss(nn.Module):
    """VGG 感知损失模块，提取 VGG16 特征并计算 L1 损失"""
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3'], device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        # 加载预训练 VGG16
        vgg = vgg16(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 定义特征提取层
        self.layers = layers
        self.layer_indices = {
            'conv1_2': 3,  # relu1_2
            'conv2_2': 8,  # relu2_2
            'conv3_3': 15, # relu3_3
        }
        self.vgg = vgg
        self.device = device

    def forward(self, y_pred, y_true):
# 单通道输入 [B, 1, H, W]，无需复制通道，直接归一化
        # 假设输入范围 [0, 1]，归一化到 ImageNet 灰度统计
        mean = torch.tensor([0.485]).view(1, 1, 1, 1).to(self.device)
        std = torch.tensor([0.229]).view(1, 1, 1, 1).to(self.device)
        y_pred = (y_pred - mean) / std
        y_true = (y_true - mean) / std

        # 修改 VGG 第一层卷积以接受单通道输入
        vgg_first_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1).to(self.device)
        # 初始化权重为 VGG 第一层 RGB 权重的平均值
        with torch.no_grad():
            vgg_first_conv.weight.copy_(self.vgg[0].weight.mean(dim=1, keepdim=True))
            vgg_first_conv.bias.copy_(self.vgg[0].bias)

        # 提取特征
        loss = 0.0
        for layer in self.layers:
            idx = self.layer_indices[layer]
            # 第一层使用单通道卷积
            pred_feat = vgg_first_conv(y_pred)
            true_feat = vgg_first_conv(y_true)
            # 后续层使用 VGG
            for i in range(2, idx + 1):  # 从第 2 层开始
                pred_feat = self.vgg[i](pred_feat)
                true_feat = self.vgg[i](true_feat)
            loss += F.l1_loss(pred_feat, true_feat)
        return loss / len(self.layers)
    

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', parameterization='eps',**kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'roofline_modules':
            from .roofline_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet) #解包unet參數，UNet(in_channels=3, out_channels=3, num_layers=4, ...)，用config中的参数初始化unet
        self.beta_schedule = beta_schedule #beta调度
        assert parameterization in ['eps', 'x0', 'v'], f"Unknown parameterization: {parameterization}"
        self.parameterization = parameterization
        self.parameterization = 'v'  # Set to v for this implementation
        
        self.vgg_loss_fn = VGGPerceptualLoss(layers=['conv1_2', 'conv2_2', 'conv3_3'], device='cuda')
    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)#partial函数，固定函数的部分参数，返回一个新的函数,这里固定了dtype和device
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance( #从计算图中分离出来，不再跟踪其梯度
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1]) #前一个gamma值，第一个gamma值为1，直到gamma_t-1

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))
        #bugger中的数据不会被更新，不会被优化器更新，但是会被保存和加载。用处：保存模型的参数，但是不会被更新
        self.register_buffer('sqrt_gammas', to_torch(np.sqrt(gammas)))
        self.register_buffer('sqrt_one_minus_gammas', to_torch(np.sqrt(1. - gammas)))

        # calculations for posterior q(x_{t-1} | x_t, x_0) 后验分布
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas) #后验方差，sigma^2
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20)))) #后验方差的对数，clip是为了防止出现负数
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas))) 
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))
        if self.parameterization == "v":
            lvlb_weights = np.ones_like(betas)  # Uniform weights for v-prediction
        else:
            lvlb_weights = betas ** 2 / (2 * posterior_variance * alphas * (1 - gammas))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', to_torch(lvlb_weights))

    def predict_start_from_noise(self, y_t, t, noise): #预测初始值y0_hat，用yt和噪声预测y0_hat
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -      #extract将a中的值取出来，采样t，reshape成b,1,1,1
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )
    
    def get_v(self, y_0, noise, t):
        return (
            extract(self.sqrt_gammas, t, y_0.shape) * noise -
            extract(self.sqrt_one_minus_gammas, t, y_0.shape) * y_0
        )

    def predict_start_from_v(self, y_t, t, v):
        return (
            extract(self.sqrt_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_one_minus_gammas, t, y_t.shape) * v
        )

    def q_posterior(self, y_0_hat, y_t, t): #p（xt-1|xt,x0_hat）#后验分布xt-1|xt,y0_hat
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):  
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device) #噪声水平累积，从gamma中提取t时刻的噪声水平
        y_0_hat = self.predict_start_from_noise(                             #预测初始值y0_hat
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)) #放入unet，得到带噪声输出
        

        if clip_denoised:   #将y0_hat的值限制在-1到1之间
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t) #预测后验分布yt-1|yt,y0_hat
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):    #正向采样，从y0采样到yt
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )
 
    @torch.no_grad()  #采样过程不需要梯度
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):  #逆向采样，从yt采样到yt-1
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t) #随机噪声
        return model_mean + noise * (0.5 * model_log_variance).exp()   #mean + noise * std. 因为model_log_variance实际上是log(std^2)，这里乘以noise的是std
    
    @torch.no_grad()
    def p_sample_ddim(self, y_t, t,t_prev, clip_denoised=True, y_cond=None, eta=0.2,control_image=None, self_cond=None): #DDIM采样，从yt采样到yt-1
        # 1. 预测 y₀_hat（初始图像估计）
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # y0_hat = self.predict_start_from_noise(
        #     y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level,control_image)
        # )
        model_output = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level, control_image,self_cond=self_cond)

        if self.parameterization == "v":
            y0_hat = self.predict_start_from_v(y_t, t, model_output)
        else:
            y0_hat = self.predict_start_from_noise(y_t, t, model_output)

        if clip_denoised:
            y0_hat.clamp_(-1., 1.)

        # 2. 根据当前时间步计算α_t和相关量
        alpha_t = extract(self.gammas, t, y_t.shape)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        # 根据公式，预测噪声ε_pred
        epsilon_pred = (y_t - sqrt_alpha_t * y0_hat) / sqrt_one_minus_alpha_t

        # 3. 获取上一时间步的α（注意处理t=0的边界情况）
        # 假设t为形状为(b,)的张量，取第一个元素判断是否为0
        if t_prev[0].item()>=0:
            alpha_prev = extract(self.gammas, t_prev, y_t.shape)
        else:
            # t==0时直接返回y0_hat（最后一步）
            return y0_hat, y0_hat

        # 4. 计算DDIM中的σ_t, 控制随机性
        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)

        # 5. DDIM更新公式：由确定性部分和随机部分构成
        y_t_prev = (torch.sqrt(alpha_prev) * y0_hat +
                    torch.sqrt(1 - alpha_prev - sigma_t**2) * epsilon_pred)
        if eta > 0:
            y_t_prev = y_t_prev + sigma_t * torch.randn_like(y_t)

        return y_t_prev, y0_hat

    # @torch.no_grad()
    # def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8): #恢复图像的完整过程
    #     b, *_ = y_cond.shape
    #     #y_cond是已知的部分或者叫做条件部分，y_t是要采样的部分，y_0是原始图像（如果有），mask是已知的部分（如果有）
    #     assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
    #     sample_inter = None if sample_num == 0 else (self.num_timesteps//sample_num) #采样间隔，每隔多少时间步评估一次
        
    #     y_t = default(y_t, lambda: torch.randn_like(y_cond)) #如果yt不存在，就用随机噪声代替，大多数情况下是这样，但是
    #     ret_arr = y_t 
    #     #采样到t=0
    #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
    #         t = torch.full((b,), i, device=y_cond.device, dtype=torch.long) #形状为b的全为i的张量
    #         y_t = self.p_sample(y_t, t, y_cond=y_cond)#逆向采样，从yt采样到yt-1
    #         if mask is not None: #在补全任务中，mask是已知的，所以需要将已知的部分保留下来
    #             y_t = y_0*(1.-mask) + mask*y_t#y0是图像补全任务中的原始图像，mask是已知的部分，y_t是采样的图像（roofdiffusion里，全部图片都被加入噪声，所以mask不可用）
    #         if sample_inter is not None and i % sample_inter == 0:
    #             ret_arr = torch.cat([ret_arr, y_t], dim=0) #将采样的图像拼接起来，方便后续的评估
    #     return y_t, ret_arr
    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8,ddim_sample_steps=50,control_image=None): #恢复图像的完整过程
        """
        使用 DDIM 采样进行图像恢复。
        参数：
            y_cond: 已知条件部分
            y_t: 初始噪声（如果没有提供，则随机初始化）
            y_0: 原始图像（用于补全任务）
            mask: 已知区域的掩码（用于补全任务）
        注意：
            self.num_timesteps 用于噪声调度表的构建，
            self.ddim_sample_steps 控制实际采样步数（比完整时间步数少）。
        """
        b, *_ = y_cond.shape
        # 构建 DDIM 采样的时间步序列，从 self.num_timesteps-1 到 0，共采样 self.ddim_sample_steps 步
        ddim_steps = ddim_sample_steps
        # 使用 linspace 生成等间隔的时间步（注意类型和设备）
        timesteps = torch.linspace(0, self.num_timesteps - 1, steps=ddim_steps, dtype=torch.long, device=y_cond.device)
        timesteps = timesteps.flip(0)  # 逆序采样，从高到低
        #timesteps_prev
        timesteps_prev = torch.cat([timesteps, torch.tensor([-1], device=y_cond.device, dtype=torch.long)]) #将-1加入到timesteps中
        timesteps_prev = timesteps_prev[1:] #去掉第一个
        # 如果没有提供 y_t，则初始化为随机噪声
        # y_t = default(y_t, lambda: torch.randn_like(y_cond))
        # ret_arr = [y_t]
        # control_image = torch.zeros_like(control_image)

        # for t_val,t_prev in zip(timesteps, timesteps_prev):
        #     t_tensor = torch.full((b,), t_val.item(), device=y_cond.device, dtype=torch.long)
        #     t_tensor_prev = torch.full((b,), t_prev.item(), device=y_cond.device, dtype=torch.long)
        #     # 使用 DDIM 采样函数 p_sample_ddim 进行采样
        #     y_t = self.p_sample_ddim(y_t, t_tensor,t_tensor_prev, y_cond=y_cond,control_image=control_image)
        #     # 如果是补全任务，保留已知区域
        #     if mask is not None and y_0 is not None:
        #         y_t = y_0 * (1. - mask) + mask * y_t
        #     ret_arr.append(y_t)
        
        # # 拼接所有采样的中间结果, ret arr will be saved as a tensor
        # ret_arr = torch.cat(ret_arr, dim=0)
        # return y_t, ret_arr
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        y_t = y_t * mask + y_0*(1. - mask) #将y_t和mask进行融合，mask是已知的部分，y_t是采样的图像
        y_cond_mask = (y_cond > -1.).to(y_cond.device).float()
        t_tensor = torch.full((b,), self.num_timesteps - 1, device=y_t.device, dtype=torch.long)
        gamma_t1 = extract(self.gammas, t_tensor, x_shape=(1, 1))
        #y_t = self.q_sample(y_0=y_cond, sample_gammas=gamma_t1.view(-1, 1, 1, 1), noise=torch.randn(y_cond.shape).to(y_cond.device))

        ret_arr = [y_t]
        y0_hat = torch.zeros_like(y_t)
        ret_y0 = [torch.zeros_like(y_t)]
        from torchvision.utils import save_image
        save_image((control_image+1.)/2., 'control_imagebefore.png')

        control_ = control_image.clone()

        noise_control = torch.randn_like(control_image)
        control_image = self.q_sample(y_0=control_image, sample_gammas=gamma_t1.view(-1, 1, 1, 1), noise=noise_control)
        control_image = control_image * y_cond_mask + y_cond * (1. - y_cond_mask)
        control_image = control_image * mask -  (1. - mask)
        control_image = torch.zeros_like(control_image) 
        from torchvision.utils import save_image
        save_image((control_image+1.)/2., 'control_image.png')

    
        for t_val, t_prev in zip(timesteps, timesteps_prev):

            t_tensor = torch.full((b,), t_val.item(), device=y_cond.device, dtype=torch.long)
            t_tensor_prev = torch.full((b,), t_prev.item(), device=y_cond.device, dtype=torch.long)
            y_t, y0_hat = self.p_sample_ddim(y_t, t_tensor, t_tensor_prev, y_cond=y_cond, control_image=control_image,self_cond=y0_hat)
            extracted_gamma = extract(self.gammas, t_tensor, x_shape=(1, 1))
            noise_control = torch.randn_like(control_image)
            control_image = self.q_sample(y_0=control_, sample_gammas=extracted_gamma.view(-1, 1, 1, 1), noise=noise_control)
            control_image = control_image * mask - (1. - mask)
            control_image = torch.zeros_like(control_image)
            y0_hat = y0_hat * mask + y_0 * (1. - mask) #将y0_hat和原始图像进行融合，mask是已知的部分，y_0是原始图像
            from torchvision.utils import save_image
            save_image((control_image+1.)/2., 'control_imagepro.png')
            if mask is not None and y_0 is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            ret_arr.append(y_t)
            ret_y0.append(y0_hat)

        ret_arr = torch.cat(ret_arr, dim=0)
        ret_y0 = torch.cat(ret_y0, dim=0)
        return y_t, ret_arr, ret_y0


    def forward(self, y_0, y_cond=None, mask=None, noise=None,control_image=None): #训练过程
        # sampling from p(gammas)
        b, *_ = y_0.shape #b是batch_size
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() #随机采样一个时间步，形状为b，
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1)) #提取t-1时刻的gamma，形状为b,-》b,1,1
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))    #提取t时刻的gamma，形状为b,-》b,1,1
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1  
        #这一步的目的是在两个连续时间步之间随机采样一个值，使得训练中使用的 gamma 参数有一定的连续性和随机性。
        sample_gammas = sample_gammas.view(b, -1)#将采样的gamma值reshape成b,1

        noise = default(noise, lambda: torch.randn_like(y_0))
        # y_noisy = self.q_sample(
        #     y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise) #加噪后的图像
        
        # if torch.sum(torch.abs(control_image)) == 0: #如果control_image是全0的张量，就不加噪声
        #     control_image = torch.zeros_like(y_0) #全0的张量
        # else:
        #     noise_of_control = torch.randn_like(control_image) 
        #     control_image = self.q_sample(
        #         y_0=control_image, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_of_control) #加噪后的控制图像
        #     control_image = control_image*mask - (1.-mask) #将控制图像和原始图像进行融合，mask是已知的部分，y_0是原始图像
        # Step 1: 判断哪些 batch 是全 0
        zero_mask = (control_image.abs().sum(dim=(1, 2, 3)) == 0)  # (B,)

        # Step 2: 准备噪声和 gamma
        noise = torch.randn_like(control_image)
        gammas = sample_gammas.view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # Step 3: 执行加噪
        q_sampled = self.q_sample(y_0=control_image, sample_gammas=gammas, noise=noise)

        # Step 4: 融合 mask
        control_image_noisy = q_sampled * mask - (1. - mask)  # (B, C, H, W)

        # Step 5: 替换那些原本是全 0 的 batch 为 0 张量
        control_image = torch.where(zero_mask.view(-1, 1, 1, 1), torch.zeros_like(y_0), control_image_noisy)

        control_image = torch.zeros_like(control_image)  # 全 0 的张量

        from torchvision.utils import save_image
        save_image((control_image+1.)/2., 'control_image.png')
        save_image((y_cond+1.)/2., 'y_cond.png')
        y_noisy = self.q_sample(y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        # Self-conditioning: Initialize y0_hat_selfcond as zeros
        y0_hat_selfcond = torch.zeros_like(y_0)  # Shape: [B, C, H, W]
        selfcond_prob = 0.5  # Probability of applying self-conditioning

        # Randomly decide whether to compute an initial prediction for self-conditioning
        if torch.rand(1).item() < selfcond_prob:
            # Compute initial prediction without self-conditioning
            with torch.no_grad():
                initial_input = torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1)
                initial_output = self.denoise_fn(initial_input, sample_gammas, control_image, self_cond=y0_hat_selfcond)
            
            # Predict y0_hat based on parameterization
            if self.parameterization == "v":
                y0_hat_selfcond = self.predict_start_from_v(y_noisy, t, initial_output)
            else:
                y0_hat_selfcond = self.predict_start_from_noise(y_noisy, t, initial_output)

        y0_hat_selfcond = y0_hat_selfcond * mask + y_0 * (1. - mask)  # Combine with original image

        model_output = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas, control_image, self_cond=y0_hat_selfcond)

        if self.parameterization == "v":
            target = self.get_v(y_0, noise, t)
            y0_hat = self.predict_start_from_v(y_noisy, t, model_output)
        else:
            target = noise
            y0_hat = self.predict_start_from_noise(y_noisy, t, model_output)


        alpha_bar = sample_gammas.view(-1)
        snr = alpha_bar / (1. - alpha_bar + 1e-8)
        gamma = 5.0
        snr_weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr
        snr_weight = snr_weight.view(-1, 1, 1, 1)

        if mask is not None:
            mask_allone = torch.ones_like(mask)
            y0_hat = y0_hat * mask + y_0 * (1. - mask) #将y0_hat和原始图像进行融合，mask是已知的部分，y_0是原始图像
            edge0_hat = (y0_hat+1.)/2.
            if hasattr(self.loss_fn, '__name__') and self.loss_fn.__name__ in ['masked_mse_loss', 'masked_l1_loss', 'combined_loss_with_masked_ssim', 'masked_l1_loss_samplewise']:
                loss_v = self.loss_fn(target, model_output, mask_allone)
                #loss_y0 = ssim_loss(mask * y_0, mask * y0_hat)
                loss_y0_CE = cross_entropy_loss_RCF(edge0_hat, (y_0+1.)/2.)
                #   print('loss_y0_CE shape:', loss_y0_CE.shape)
                #loss_y0_dice = dice_loss(mask * y0_hat, mask * y_0) not good, details are lost
                #loss_ssim = ssim_loss(mask * y0_hat, mask * y_0) # not suitable for big range of noise
                loss_tv = tv_loss(mask * y0_hat)
                #loss_vgg = self.vgg_loss_fn(y0_hat*mask, y_0*mask) * 0.2
                #loss_tversky = tversky_loss(mask * y0_hat, mask * y_0)
                #loss_l1 = self.loss_fn(mask * y_0, mask * edge0_hat) * 0.2
                loss = (loss_v * snr_weight.view(-1)).mean() + (snr_weight.view(-1)* loss_y0_CE).mean() # + (snr_weight.view(-1) * loss_l1).mean() #+ loss_vgg.mean()
            else:
                loss = self.loss_fn(mask * target, mask * model_output)
        else:
            loss = self.loss_fn(target, model_output)

        return loss, y0_hat

        # if mask is not None:
        #     noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,control_image)
        #     #mask存在时，只保留mask部分
        #     # from torchinfo import summary
        #     # params = {}#unet.copy()
        #     # params['input_size'] = (4,2,256,256)
        #     # params['gammas'] = sample_gammas
        #     # summary(self.denoise_fn, **params, col_names=("input_size", "output_size", "num_params"), depth=10)

        #     if hasattr(self.loss_fn, '__name__') and self.loss_fn.__name__ in ['masked_mse_loss', 'masked_l1_loss', 'combined_loss_with_masked_ssim']:
        #         loss = self.loss_fn(mask*noise, mask*noise_hat, mask) # might not be necessary 
        #     else:
        #         loss = self.loss_fn(mask*noise, mask*noise_hat)
        # else:
        #     noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas,control_image) #从y_cond和y_noisy中去噪声，通过unet
        #     #mask不存在时，直接去噪声，不需要保留mask部分。也就是说此时的y_noisy在mask外的部分也有噪声
        #     loss = self.loss_fn(noise, noise_hat)
        # return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val): #如果val存在，返回val，否则返回d
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)): #a是列表，t是时间步，x_shape是张量的形状
    # 
    b, *_ = t.shape #星号表示解包成一个列表，b是t的第一个维度，batch_size形状（b，）
    out = a.gather(-1, t) #在a的最后一个维度上，根据t的索引值，将a中的值取出来。与t形状相同
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) #(1,) * (len(x_shape) - 1)-》(1,1,1)，星号表示解包成一个列表并传入函数，等价于reshape(b,1,1,1)

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def ssim_loss(output, target):
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(output.device)
    ssim_value = ssim(output, target)
    return 1.0 - ssim_value


def cross_entropy_loss_RCF(prediction_, labelf_, beta=0.9):
    #add gaussian  blur to lebelf


    prediction = prediction_.clamp(1e-8, 1 - 1e-8) #将预测值限制在1e-8到1-1e-8之间
    #prediction[prediction < 0.03] = 0 #将预测值小于1e-8的值设为1e-8
    # prediction[prediction > 0.1] = 1 #将预测值大于1-1e-8的值设为1-1e-8

    labelf = labelf_.clone() # (B, 1, H, W)
    labelf[labelf >0.1] = 1.0 #将标签值大于-1+1e-5的值设为1.0
    labelf[labelf <= 0.1] = 0.0 #将标签值小于等于-1+1e-5的值设为0.0

    from torchvision.transforms import GaussianBlur
    gaussian_blur = GaussianBlur(kernel_size=3, sigma=1.0)
    labelf_blur = gaussian_blur(labelf)
    labelf = (labelf_blur + labelf).clamp(0, 1) #将标签值限制在0到1之间
    from torchvision.utils import save_image
    save_image(prediction, 'prediction.png')
    save_image(labelf, 'labelf.png')
    

    # label = labelf.long()
    label = labelf # (B, 1, H, W)
    mask = labelf.clone() # (B, 1, H, W)

    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float() 

    mask_temp = (labelf > 0) & (labelf <= 0.3) #范围是 (0, 0.3]
    mask[mask_temp] = 0 #将范围 (0, 0.3] 的值设为 0

    #edge_mask, non_edge_mask = sobel_edge_detection(prediction, threshold=0.1) #对单通道图像应用Sobel边缘检测，返回边缘掩码（1表示边缘，0表示非边缘）

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative) #负向样本的权重
    mask[label == 0] = beta* num_positive / (num_positive + num_negative) #正向样本的权。正样本权重与负样本数量成正比，负样本权重与正样本数量成正比，平衡正负样本的不均等。适当负样本权重，因为负样本大（非边缘）数量多，正样本小（边缘）数量少，正样本权重大于负样本权重
    #mask[(edge_mask == 0) & (label == 0)] = 0.3 * num_negative / (num_positive + num_negative) #边缘区域的权重

    # mask[label == 2] = 0
    cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='none')
    return cost.mean([1, 2, 3]) #在c，h,w维度上求均值，返回每个batch的平均损失，形状为 (B, 1)

def dice_loss(prediction, labelf, smooth=1.0):
    """
    Dice Loss，优化边缘区域重叠
    Args:
        prediction: (B, 1, H, W), sigmoid后的概率
        labelf: (B, 1, H, W), 标签值 in {0, 1} 或 (0, 0.3]
        smooth: 平滑项
    """
    prediction = prediction.contiguous().view(prediction.size(0), -1)  # (B, N)
    labelf = labelf.contiguous().view(labelf.size(0), -1)  # (B, N)

    # 忽略 (0, 0.3]
    mask = (labelf == 0) | (labelf == 1)
    prediction = prediction * mask.float()
    labelf = labelf * mask.float()

    intersection = (prediction * labelf).sum(dim=1)
    union = prediction.sum(dim=1) + labelf.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth + 1e-8)
    return 1 - dice.mean()


def sobel_edge_detection(image, threshold=0.1):
    """对单通道图像应用Sobel边缘检测，返回边缘掩码（1表示边缘，0表示非边缘）"""
    # image: [B, 1, H, W]
    # Sobel核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    
    # 计算梯度
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # 归一化梯度到[0, 1]
    grad_magnitude = grad_magnitude / grad_magnitude.max()
    
    # 边缘掩码：梯度大于阈值为边缘（1），否则为非边缘（0）
    edge_mask = (grad_magnitude > threshold).float()
    from torchvision.utils import save_image
    save_image(edge_mask, 'edge_mask.png')

    return edge_mask, 1 - edge_mask  # 返回边缘掩码和非边缘掩码


from torchmetrics.image import StructuralSimilarityIndexMeasure

def compute_ssim(img1, img2, window_size=5, data_range=1.0, return_loss=True):
    """
    使用 torchmetrics 计算两张图像的 SSIM（结构相似性指数）。

    参数：
        img1 (torch.Tensor): 输入图像1，形状为 [B, C, H, W]，范围通常为 [0, 1]
        img2 (torch.Tensor): 输入图像2，形状为 [B, C, H, W]，范围通常为 [0, 1]
        window_size (int): 滑动窗口大小，推荐 3 或 5 用于二值化图像
        data_range (float): 图像像素值的范围（默认 1.0，适用于 [0, 1] 图像）
        return_loss (bool): 如果为 True，返回 SSIM 损失 (1 - SSIM)；否则返回 SSIM 值

    返回：
        torch.Tensor: SSIM 值或 SSIM 损失, 形状为 [B]，表示每个图像对的 SSIM 值
    """
    # 输入检查
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if len(img1.shape) != 4:
        raise ValueError("Expected 4D input tensors [B, C, H, W].")

    # 初始化 SSIM 模块
    ssim_metric = StructuralSimilarityIndexMeasure(
        kernel_size=window_size,  # 窗口大小
        data_range=data_range,    # 像素值范围
        return_full_image=False   # 返回平均 SSIM 值
    ).to(img1.device)

    # 计算 SSIM
    ssim_value = ssim_metric(img1, img2)

    # 返回 SSIM 值或损失
    if return_loss:
        return 1 - ssim_value
    return ssim_value

def tv_loss(img, weight=1.0, normalize=True):
    """
    计算总变差损失（TV Loss），鼓励图像平滑和二值化。

    参数：
        img (torch.Tensor): 输入图像，形状为 [B, C, H, W]，通常范围 [0, 1]
        weight (float): TV Loss 的权重，控制平滑强度
        normalize (bool): 是否对损失进行归一化（按图像尺寸）

    返回：
        torch.Tensor: TV Loss 值
    """
    # 输入检查
    if len(img.shape) != 4:
        raise ValueError("Expected 4D input tensor [B, C, H, W]")

    # 计算水平和垂直方向的像素差异
    h_diff = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])  # 垂直差异
    w_diff = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])  # 水平差异

    # 求和并归一化
    loss = h_diff.sum() + w_diff.sum()
    if normalize:
        batch, _, height, width = img.shape
        loss = loss / (batch * height * width)  # 按图像尺寸归一化

    return weight * loss

