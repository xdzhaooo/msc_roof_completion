import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from torch.cuda.amp import custom_bwd, custom_fwd
from .denoising_diffusion_pytorch.encoder_decoder import DiagonalGaussianDistribution
import torch.nn.functional as F
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        #self.Roofdenoise_fn = UNet(**unet) #解包unet參數，UNet(in_channels=3, out_channels=3, num_layers=4, ...)，用config中的参数初始化unet
        self.RoofLinedenoise_fn = UNet(**unet) #解包unet參數，UNet(in_channels=3, out_channels=3, num_layers=4, ...)，用config中的参数初始化unet
        self.beta_schedule = beta_schedule #beta调度
        '''____ latent diffusion parameters ________________'''
        self.scale_by_std = True #latent diffusion的scale_by_std
        self.scale_factor = 0.3 #latent diffusion的scale_factor
        self.scale_by_softsign = False #latent diffusion的scale_by_softsign
        self.eps = 1e-4 #latent diffusion的eps

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train',accelerator=None): #设置新的噪声调度表
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            self.accelerator = None
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

        # calculations for posterior q(x_{t-1} | x_t, x_0) 后验分布
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas) #后验方差，sigma^2
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20)))) #后验方差的对数，clip是为了防止出现负数
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas))) 
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise): #预测初始值y0_hat，用yt和噪声预测y0_hat
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -      #extract将a中的值取出来，采样t，reshape成b,1,1,1
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def predict_start_from_noise_with_gamma(self,y_t, gamma, noise):
        """
        使用提供的 gamma 值（\bar{\alpha}_t）预测初始数据 \hat{y}_0。
        参数：
            y_t: 加噪数据，形状为 (batch_size, channels, height, width)
            gamma: \bar{\alpha}_t 值，形状为 (batch_size,) 或标量，表示累积 alpha
            noise: 预测的噪声，形状与 y_t 相同
        返回：
            y0_hat: 预测的初始数据，形状与 y_t 相同
        """
        # 确保 gamma 是张量并移动到 y_t 的设备
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, dtype=torch.float32, device=y_t.device)
        
        # 如果 gamma 是标量，扩展为 (batch_size,)
        if gamma.dim() == 0:
            gamma = gamma.expand(y_t.shape[0])
        
        # 确保 gamma 形状为 (batch_size,)
        gamma = gamma.view(-1)
        
        # 计算 sqrt_recip_gammas = 1 / sqrt(gamma)
        sqrt_recip_gammas = 1.0 / torch.sqrt(gamma)
        
        # 计算 sqrt_recipm1_gammas = sqrt((1 - gamma) / gamma)
        sqrt_recipm1_gammas = torch.sqrt((1.0 - gamma) / gamma)
        
        # 调整形状以匹配 y_t 的形状，例如 (batch_size, 1, 1, 1)
        sqrt_recip_gammas = sqrt_recip_gammas.view(-1, 1, 1, 1)
        sqrt_recipm1_gammas = sqrt_recipm1_gammas.view(-1, 1, 1, 1)
        
        # 计算 \hat{y}_0 = (y_t - sqrt(1 - \bar{\alpha}_t) * noise) / sqrt(\bar{\alpha}_t)
        y0_hat = sqrt_recip_gammas * y_t - sqrt_recipm1_gammas * noise
        
        return y0_hat
    

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
    def p_sample_ddim_rl(self, y_t, t,t_prev, clip_denoised=True, y_cond=None, eta=1.0,control_image=None):  # 使用ddim进行逆向采样
        # 1. 预测 y₀_hat（初始图像估计）
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.RoofLinedenoise_fn(y_t, noise_level,torch.cat([y_cond,control_image], dim=1)) #从y_cond和y_t中去噪声，通过unet
        )

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
            return y0_hat,y0_hat

        # 4. 计算DDIM中的σ_t, 控制随机性
        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)

        # 5. DDIM更新公式：由确定性部分和随机部分构成
        y_t_prev = (torch.sqrt(alpha_prev) * y0_hat +
                    torch.sqrt(1 - alpha_prev - sigma_t**2) * epsilon_pred)
        if eta > 0:
            y_t_prev = y_t_prev + sigma_t * torch.randn_like(y_t)

        return y_t_prev,y0_hat
    
    @torch.no_grad()
    def p_sample_ddim_r(self, y_t, t,t_prev, clip_denoised=True, y_cond=None, eta=0.0,control_image=None):  # 使用ddim进行逆向采样
        # 1. 预测 y₀_hat（初始图像估计）
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.Roofdenoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level,control_image)
        )

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
            return y0_hat,y0_hat

        # 4. 计算DDIM中的σ_t, 控制随机性
        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)

        # 5. DDIM更新公式：由确定性部分和随机部分构成
        y_t_prev = (torch.sqrt(alpha_prev) * y0_hat +
                    torch.sqrt(1 - alpha_prev - sigma_t**2) * epsilon_pred)
        if eta > 0:
            y_t_prev = y_t_prev + sigma_t * torch.randn_like(y_t)

        return y_t_prev,y0_hat

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
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8,ddim_sample_steps=1000,roofline_image=None): #恢复图像的完整过程
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
        device = self.accelerator.device if self.accelerator is not None else y_cond.device
        b, *_ = y_cond.shape
        # 构建 DDIM 采样的时间步序列，从 self.num_timesteps-1 到 0，共采样 self.ddim_sample_steps 步
        ddim_steps = ddim_sample_steps
        # 使用 linspace 生成等间隔的时间步（注意类型和设备）
        timesteps = torch.linspace(0, self.num_timesteps - 1, steps=ddim_steps, dtype=torch.long, device=y_cond.device)
        timesteps = timesteps.flip(0)  # 逆序采样，从高到低
        #timesteps_prev
        timesteps_prev = torch.cat([timesteps, torch.tensor([-1], device=device, dtype=torch.long)]) #将-1加入到timesteps中
        timesteps_prev = timesteps_prev[1:] #去掉第一个
        # 如果没有提供 y_t，则初始化为随机噪声
        y_t_roof = default(y_t, lambda: torch.randn_like(y_cond))
        y_t_roofline = default(y_t, lambda: torch.randn_like(roofline_image))
        y_0_hat_roofline = torch.zeros_like(roofline_image) #初始化y0_hat_roofline为0
        ret_arr_roof = [y_t_roof] #保存采样的图像
        ret_arr_roofline = [y_t_roofline]
        ret_arr_roofline_y0 = [y_0_hat_roofline] #保存采样的图像

        for t_val,t_prev in zip(timesteps, timesteps_prev):
            t_tensor = torch.full((b,), t_val.item(), device=device, dtype=torch.long)
            t_tensor_prev = torch.full((b,), t_prev.item(), device=device, dtype=torch.long)
            # 使用 DDIM 采样函数 p_sample_ddim 进行采样
            y_t_roofline,y_0_hat_roofline = self.p_sample_ddim_rl(y_t_roofline, t_tensor,t_tensor_prev, y_cond=y_cond,eta=0.0,control_image=y_t_roof) #roofline的采样
            y_t_roof,_ = self.p_sample_ddim_r(y_t_roof, t_tensor,t_tensor_prev, y_cond=y_cond,control_image=y_0_hat_roofline)

            # 如果是补全任务，保留已知区域
            if mask is not None and y_0 is not None:
                y_t_roof = y_0 * (1. - mask) + mask * y_t_roof
                y_t_roofline = roofline_image * (1. - mask) + mask * y_t_roofline
            ret_arr_roof.append(y_t_roof) #保存采样的图像
            ret_arr_roofline.append(y_t_roofline)
            ret_arr_roofline_y0.append(y_0_hat_roofline)
        
        # 拼接所有采样的中间结果, ret arr will be saved as a tensor
        ret_arr_roof = torch.cat(ret_arr_roof, dim=0)
        ret_arr_roofline = torch.cat(ret_arr_roofline, dim=0)
        ret_arr_roofline_y0 = torch.cat(ret_arr_roofline_y0, dim=0)
        return y_t_roof, ret_arr_roof, ret_arr_roofline, ret_arr_roofline_y0, y_0_hat_roofline


    def forward(self, y_0, y_cond=None, mask=None, noise=None,roofline_image=None,roofline_noise=None): #训练过程
        '''
        y_0: heightmap
        y_cond: corrupted heightmap
        mask: mask for corrupted heightmap
        noise: noise for y_0, default is None, if None, use random noise
        roofline_image: roofline image for roof completion, default is None, if None, use y_cond'''
        # sampling from p(gammas)
        b, *_ = y_0.shape #b是batch_size
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() #随机采样一个时间步，形状为b，
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1)) #提取t-1时刻的gamma，形状为b,-》b,1,1
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))    #提取t时刻的gamma，形状为b,-》b,1,1
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1  
        #这一步的目的是在两个连续时间步之间随机采样一个值，使得训练中使用的 gamma 参数有一定的连续性和随机性。
        sample_gammas = sample_gammas.view(b, -1)#将采样的gamma值reshape成b,1

        noise = default(noise, lambda: torch.randn_like(y_0)) #heightmap的噪声
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise) #加噪后的图像
        
        noise_roofline = default(roofline_noise, lambda: torch.randn_like(roofline_image)) #roofline的噪声
        y_roofline_noisy = self.q_sample(
            y_0=roofline_image, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_roofline) #加噪后的roofline图像

        if mask is not None:
            #noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,control_image)
            roofline_noise_hat = self.RoofLinedenoise_fn(torch.cat([y_cond, y_roofline_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,y_noisy) #从y_cond和y_noisy中去噪声，通过unet
            roofline_y0_hat = self.predict_start_from_noise_with_gamma(y_roofline_noisy, sample_gammas, roofline_noise_hat) #预测y0_hat
            roofline_y0_hat = mask*roofline_y0_hat 
            roof_noise_hat = self.Roofdenoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,roofline_y0_hat) #从y_cond和y_noisy中去噪声，通过unet
            #mask存在时，只保留mask部分
            # from torchinfo import summary
            # params = {}#unet.copy()
            # params['input_size'] = (4,2,256,256)
            # params['gammas'] = sample_gammas
            # summary(self.denoise_fn, **params, col_names=("input_size", "output_size", "num_params"), depth=10)

            if hasattr(self.loss_fn, '__name__') and self.loss_fn.__name__ in ['masked_mse_loss', 'masked_l1_loss', 'combined_loss_with_masked_ssim']:
                #loss = self.loss_fn(mask*noise, mask*noise_hat, mask) # might not be necessary 
                loss_roof_noise = self.loss_fn(mask*noise, mask*roof_noise_hat, mask) # might not be necessary
                loss_roofline_noise = self.loss_fn(mask*noise_roofline, mask*roofline_noise_hat, mask) # might not be necessary
                loss_roofline_recon = self.loss_fn(mask*roofline_image, mask*roofline_y0_hat, mask) # might not be necessary
                loss = loss_roof_noise + loss_roofline_noise + loss_roofline_recon
            else:
                #loss = self.loss_fn(mask*noise, mask*noise_hat)
                loss_roof_noise = self.loss_fn(mask*noise, mask*roof_noise_hat, mask) # might not be necessary
                loss_roofline_noise = self.loss_fn(mask*noise_roofline, mask*roofline_noise_hat, mask) # might not be necessary
                loss_roofline_recon = self.loss_fn(mask*roofline_image, mask*roofline_y0_hat, mask) # might not be necessary
                loss = loss_roof_noise + loss_roofline_noise + loss_roofline_recon
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas) #从y_cond和y_noisy中去噪声，通过unet
            #mask不存在时，直接去噪声，不需要保留mask部分。也就是说此时的y_noisy在mask外的部分也有噪声
            loss = self.loss_fn(noise, noise_hat)
        return loss,{"loss_roof_noise":loss_roof_noise,"loss_roofline_noise":loss_roofline_noise,"loss_roofline_recon":loss_roofline_recon} ,roofline_y0_hat#返回损失和其他信息，方便后续的评估和保存模型


    """------------------------------------------------latent diffusion functions--------------------------------------------------"""
    @torch.no_grad()
    def on_train_batch_start(self, roofline_image): #
        # only for the first batch
        if self.scale_by_std and (not self.scale_by_softsign):
            if not self.default_scale:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                print("### USING STD-RESCALING ###")
                encoder_posterior = self.first_stage_model.encode(roofline_image) #编码器的输出
                z = self.get_first_stage_encoding(encoder_posterior)
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
                # print("### USING STD-RESCALING ###")
            else:
                print(f'### USING DEFAULT SCALE {self.scale_factor}')
        else:
            print(f'### USING SOFTSIGN SCALE !')
    
    def set_autoencoder(self, autoencoder):
        assert isinstance(autoencoder, torch.nn.Module), 'autoencoder must be a torch.nn.Module'
        assert hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode'), 'autoencoder must have encode and decode functions'
        self.first_stage_model = autoencoder.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # return self.scale_factor * z.detach() + self.scale_bias
        return z.detach()

    @torch.no_grad()
    def encode_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        # assert 'image' in self.input_keys
        # # if len(self.input_keys) > len(batch.keys()):
        # #     x, cond, *_ = batch.values()
        # # else:
        # #     x, cond = batch.values()
        # x = batch['image']
        # cond = batch['cond'] if 'cond' in batch else None
        # z = self.first_stage_model.encode(x)
        # # print('zzzz', z.shape)
        # z = self.get_first_stage_encoding(z)
        # out = [z, cond, x]
        # if return_first_stage_outputs:
        #     xrec = self.first_stage_model.decode(z)
        #     out.extend([x, xrec])
        # if return_original_cond:
        #     out.append(cond)
        # return out
        z = self.first_stage_model.encode(batch) #编码器的输出
        z = self.get_first_stage_encoding(z) #获取编码器的输出
        z_rec = self.first_stage_model.decode(z) #解码器的输出
        return z, z_rec
    
    def forward_latent(self, y_0, y_cond=None, mask=None, noise=None,roofline_image=None,roofline_noise=None): #训练过程sssssssssssssssssssssssssss
        roofline_encode,roofline_rec = self.encode_input(roofline_image)
        # print(roofline_encode.shape)
        # print(roofline_rec.shape)
        from torchvision.utils import save_image
        save_image(roofline_rec, 'debug/roofline_rec1.png', nrow=8)
        save_image(roofline_image, 'debug/roofline_image5.png', nrow=8)
        #print("range of roofline)image:", roofline_image.min(), roofline_image.max())
        #print("roofline_img[0]:", roofline_image[0])

        # print(_[0].shape)
        if self.scale_by_softsign:
            roofline_encode = F.softsign(roofline_encode)
        elif self.scale_by_std:
            roofline_encode = self.scale_factor * roofline_encode


        b, *_ = y_0.shape #b是batch_size
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() #随机采样一个时间步，形状为b，
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1)) #提取t-1时刻的gamma，形状为b,-》b,1,1
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))    #提取t时刻的gamma，形状为b,-》b,1,1
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1  
        #这一步的目的是在两个连续时间步之间随机采样一个值，使得训练中使用的 gamma 参数有一定的连续性和随机性。
        sample_gammas = sample_gammas.view(b, -1)#将采样的gamma值reshape成b,1

        noise = default(noise, lambda: torch.randn_like(y_0)) #heightmap的噪声
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise) #加噪后的图像
        
        noise_roofline = default(roofline_noise, lambda: torch.randn_like(roofline_encode)) #roofline的噪声
        y_roofline_noisy = self.q_sample(
            y_0=roofline_encode, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_roofline) #加噪后的roofline图像

        #仅训练roofline的时候
        y_noisy = torch.zeros_like(y_noisy)
        
        
        if mask is not None:
            #noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,control_image)
            #roofline_noise_hat = self.RoofLinedenoise_fn(torch.cat([y_cond, y_roofline_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,y_noisy) #从y_cond和y_noisy中去噪声，通过unet
            roofline_noise_hat = self.RoofLinedenoise_fn(y_roofline_noisy, sample_gammas, torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1))
            roofline_y0_hat = self.predict_start_from_noise_with_gamma(y_roofline_noisy, sample_gammas, roofline_noise_hat) #预测y0_hat
            #roof_noise_hat = self.Roofdenoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,roofline_y0_hat) #从y_cond和y_noisy中去噪声，通过unet
            #print("range of roofline_y0_hat:", roofline_y0_hat.min(), roofline_y0_hat.max())
            #print("range of roofline_noise_hat:", roofline_noise_hat.min(), roofline_noise_hat.max())

            # eps = self.eps
            loss_dict = {}
            prefix = 'train'

            target2 = noise_roofline
            target3 = roofline_encode

            loss_simple = 0. 
            loss_vlb = 0.
            # 初始化两个损失变量：
            # - loss_simple: 简单损失，监督 C_pred 和 noise_pred
            # - loss_vlb: 变分下界损失，监督 x_rec 和不确定性蒸馏

            t = t.float() / self.num_timesteps * (1.0 - self.eps) + self.eps #  # 归一化时间步 t，范围 [eps, 1.0]
            #simple_weight2 = (2 - t).sqrt() / (1 - t + self.eps).sqrt()
            #simple_weight2 = (1-sample_gammas).sqrt() 
            simple_weight2 = (1+sample_gammas).sqrt() / (sample_gammas + self.eps).sqrt() # 计算简单权重，形状 (B, 1)
            # 定义时间步相关的权重，用于加权 C_pred 和 noise_pred 的损失
            # simple_weight2: sqrt(2 - t) / sqrt(1 - t + eps)，随 t 减小而增加
            # eps 是小常数，防止除零

            # if self.weighting_loss:
            #     simple_weight1 = 2 * torch.exp(1 - t)
            #     simple_weight2 = torch.exp(torch.sqrt(t))
            #     if self.cfg.model_name == 'ncsnpp9':
            #         simple_weight1 = (t + 1) / t.sqrt()
            #         simple_weight2 = (2 - t).sqrt() / (1 - t + self.eps).sqrt()
            # else:
            #     simple_weight1 = 1
            #     simple_weight2 = 1
            #print(('simple_weight2:', simple_weight2))
            #loss_simple += simple_weight2 * self.get_loss(roofline_noise_hat, target2, mean=False).mean([1, 2, 3])
            loss_simple += self.get_loss(roofline_noise_hat, target2, mean=False).mean([1, 2, 3]) # : (B, 1)


            #print(('loss_simple values1:', loss_simple))
            
            # 计算 loss_simple 的核心部分：
            # - get_loss(C_pred, target1): 计算 C_pred 和 C 的损失（L1 或 L2，取决于 self.loss_type）
            # - mean([1, 2, 3]): 对空间维度 (C, H, W) 求均值，得到 (B,) 形状
            # - simple_weight1: 加权 C_pred 损失
            # - 同理处理 noise_pred 和 target2
            # 结果累加到 loss_simple，形状 (B,)

            # loss_simple += self.Dice_Loss(C_pred, target1) * simple_weight1

            # 如果启用 L1 损失（self.use_l1=True）：
            # - 计算 C_pred 和 noise_pred 的 L1 损失（绝对值误差）
            # - 同样加权并均值，累加到 loss_simple
            # - 除以 2，平衡 L1 和 L2 损失的贡献


            #rec_weight = (1 - t.reshape(b, 1)) ** 2
            rec_weight = (sample_gammas)**2 # 计算重建损失的权重，形状 (B, 1)
            # 定义重建损失的权重 rec_weight，形状 (B, 1)
            # rec_weight = (1 - t)^2，随 t 减小（去噪后期）权重增加
            # 确保早期高噪声时间步的损失贡献较小
            # rec_weight = 1 - t.reshape(C.shape[0], 1)  # (B, 1)
            loss_simple = loss_simple.mean()
            loss_dict.update({f'{prefix}/loss_simple': loss_simple})

            loss_vlb += torch.abs(roofline_y0_hat - target3).mean([1, 2, 3]) * rec_weight # : (B, 1)
            # loss_vlb += self.Dice_Loss(x_rec, target3) * rec_weight

            # loss_vlb = loss_vlb
            loss_vlb = loss_vlb.mean()


            with torch.no_grad():
                # 如果启用不确定性蒸馏（use_disloss=True）：
                #print("roofline_y0_hat range:", roofline_y0_hat.min(), roofline_y0_hat.max())
                #print("roofline_image range:", roofline_image.min(), roofline_image.max())
                
                edge_rec = self.first_stage_model.decode(roofline_y0_hat / self.scale_factor) #从潜在空间解码，x_rec 除以 scale_factor 调整尺度
                #print("range of edge_rec:", edge_rec.min(), edge_rec.max())
                
                edge_rec = unnormalize_to_zero_to_one(edge_rec)#归一化
                #print("range of edge_rec after unnormalize:", edge_rec.min(), edge_rec.max())
                edge_rec = torch.clamp(edge_rec, min=0., max=1.) # B, 1, 320, 320
                #edge_rec = edge_rec * mask 
                # print("range of edge_rec after clamp:", edge_rec.min(), edge_rec.max())
                # print("range of roofline_image after normalize:", roofline_image.min(), roofline_image.max())

            loss_tmp = self.cross_entropy_loss_RCF(edge_rec, (roofline_image+1)/2) * rec_weight  # B, 1
            # 计算加权交叉熵损失：
            # - cross_entropy_loss_RCF 计算 edge_rec 与真实边缘图的 WCE 损失
            # - kwargs['edge'] 归一化到 [0, 1]，值为真实边缘概率
            # - 乘以 rec_weight 加权，loss_tmp 形状 (B,)
            loss_ce = SpecifyGradient.apply(roofline_y0_hat, loss_tmp.mean()) #输出的是纯1占位符，目的是为了构建计算图
            # 应用 SpecifyGradient： 
            # - x_rec: 潜在表示 \hat{z}_0
            # - loss_tmp.mean(): 标量 WCE 损失
            # - SpecifyGradient 将 \partial L_{wce} / \partial edge_rec 直接作为 x_rec 的梯度
            # - 跳过解码器梯度传播，实现不确定性蒸馏
            # print(loss_ce.shape)
            # print(loss_vlb.shape)
            loss_vlb += loss_ce.mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

            loss = loss_simple + loss_vlb
            loss_dict.update({f'{prefix}/loss': loss})



        return loss_simple, loss_dict,edge_rec #返回损失和其他信息，方便后续的评估和保存模型    
    
    def restoration_latent(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8,ddim_sample_steps=50,roofline_image=None): #恢复图像的完整过程
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
        device = self.accelerator.device if self.accelerator is not None else y_cond.device
        b, *_ = y_cond.shape
        # 构建 DDIM 采样的时间步序列，从 self.num_timesteps-1 到 0，共采样 self.ddim_sample_steps 步
        ddim_steps = ddim_sample_steps
        # 使用 linspace 生成等间隔的时间步（注意类型和设备）
        timesteps = torch.linspace(0, self.num_timesteps - 1, steps=ddim_steps, dtype=torch.long, device=y_cond.device)
        timesteps = timesteps.flip(0)  # 逆序采样，从高到低
        #timesteps_prev
        timesteps_prev = torch.cat([timesteps, torch.tensor([-1], device=device, dtype=torch.long)]) #将-1加入到timesteps中
        timesteps_prev = timesteps_prev[1:] #去掉第一个
        # 如果没有提供 y_t，则初始化为随机噪声

        latent_shape = (b, 3,32, 32)
        y_t_roof = default(y_t, lambda: torch.randn_like(y_cond))
        # y_t_roofline = default(y_t, lambda: torch.randn_like(roofline_image))
        # y_0_hat_roofline = torch.zeros_like(roofline_image) #初始化y0_hat_roofline为0
        y_t_roofline = default(y_t, lambda: torch.randn(latent_shape, device=device))
        #y_t_roofline,_ = self.encode_input(roofline_image) #编码器的输出


        # y_t_roofline,_ = self.encode_input(roofline_image)
        # y_t_roofline += torch.randn_like(y_t_roofline) 

        y_0_hat_roofline = torch.zeros(latent_shape, device=device)
        ret_arr_roof = [y_t_roof] #保存采样的图像
        ret_arr_roofline = [1. / self.scale_factor * y_t_roofline.detach()]
        ret_arr_roofline_y0 = [y_0_hat_roofline] #保存采样的图像

        y_t_roof = torch.zeros_like(y_cond, device=device) #将y_t_roofline初始化为0
        for t_val,t_prev in zip(timesteps, timesteps_prev):
            t_tensor = torch.full((b,), t_val.item(), device=device, dtype=torch.long)
            t_tensor_prev = torch.full((b,), t_prev.item(), device=device, dtype=torch.long)
            # 使用 DDIM 采样函数 p_sample_ddim 进行采样
            roof_noise = torch.randn_like(y_0) #roofline的噪声
            #y_t_roof = self.q_sample(y_0 = y_0, sample_gammas=t_tensor.view(-1, 1, 1, 1), noise=roof_noise)

            y_t_roofline,y_0_hat_roofline = self.p_sample_ddim_rl(y_t_roofline, t_tensor,t_tensor_prev, y_cond=y_cond,eta=0.0,control_image=y_t_roof) #roofline的采样
            #y_t_roof,_ = self.p_sample_ddim_r(y_t_roof, t_tensor,t_tensor_prev, y_cond=y_cond,control_image=y_0_hat_roofline)
            # 如果是补全任务，保留已知区域
            if mask is not None and y_0 is not None:
                y_t_roof = y_0 * (1. - mask) + mask * y_t_roof
                #y_t_roofline = roofline_image * (1. - mask) + mask * y_t_roofline
            ret_arr_roof.append(y_t_roof) #保存采样的图像
            #ret_arr_roofline.append(y_t_roofline)
            ret_arr_roofline.append(1. / self.scale_factor * y_t_roofline.detach())
            ret_arr_roofline_y0.append(y_0_hat_roofline)
        
        
        if self.scale_by_std:
            y_t_roofline = 1. / self.scale_factor * y_t_roofline.detach()
        elif self.scale_by_softsign:
            y_t_roofline = y_t_roofline / (1 - y_t_roofline.abs())
            y_t_roofline = y_t_roofline.detach()
        

        y_t_roofline = self.first_stage_model.decode(y_t_roofline) #解码器的输出
        y_t_roofline = unnormalize_to_zero_to_one(y_t_roofline)
        y_t_roofline = torch.clamp(y_t_roofline, min=0., max=1.) #将y_t_roofline的值限制在0到1之间

        for i in range(len(ret_arr_roofline_y0)):
            ret_arr_roofline_y0[i] = self.first_stage_model.decode(ret_arr_roofline_y0[i])
            ret_arr_roofline_y0[i] = unnormalize_to_zero_to_one(ret_arr_roofline_y0[i])
            ret_arr_roofline_y0[i] = torch.clamp(ret_arr_roofline_y0[i], min=0., max=1.)

            ret_arr_roofline[i] = self.first_stage_model.decode(ret_arr_roofline[i])
            ret_arr_roofline[i] = unnormalize_to_zero_to_one(ret_arr_roofline[i])
            ret_arr_roofline[i] = torch.clamp(ret_arr_roofline[i], min=0., max=1.)
        
        # 拼接所有采样的中间结果, ret arr will be saved as a tensor
        ret_arr_roof = torch.cat(ret_arr_roof, dim=0)
        ret_arr_roofline = torch.cat(ret_arr_roofline, dim=0)
        ret_arr_roofline_y0 = torch.cat(ret_arr_roofline_y0, dim=0)
        print("range of y_t_roof:", y_t_roof.min(), y_t_roof.max())
        print("range of y_t_roofline:", y_t_roofline.min(), y_t_roofline.max())
        print("range of ret_arr_roof:", ret_arr_roof.min(), ret_arr_roof.max())
        print("range of ret_arr_roofline:", ret_arr_roofline.min(), ret_arr_roofline.max())
        print("range of ret_arr_roofline_y0:", ret_arr_roofline_y0.min(), ret_arr_roofline_y0.max())
        #dadfd
        return y_t_roof, ret_arr_roof, ret_arr_roofline, ret_arr_roofline_y0, y_t_roofline
    

    def cross_entropy_loss_RCF(self, prediction, labelf, beta=1.1):
        #print("range of prediction", prediction.min(), prediction.max(),"dfasdfasdsdsdsdsdsdsdsdsdsdsdsdddddddddddddddddddddddddddddddddddddddddddddd")
        # label = labelf.long()
        label = labelf # (B, 1, H, W)
        mask = labelf.clone() # (B, 1, H, W)

        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float() #不确定像素

        mask_temp = (label > 0) & (label <= 0.05) #范围是 (0, 0.3]
        mask[mask_temp] = 0. #将范围 (0, 0.3] 的值设为 0, 也就是将不确定像素设为0

        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative) #负向样本的权重
        mask[label == 0] = beta * num_positive / (num_positive + num_negative) #正向样本的权。正样本权重与负样本数量成正比，负样本权重与正样本数量成正比，平衡正负样本的不均等。适当负样本权重，因为负样本大（非边缘）数量多，正样本小（边缘）数量少，正样本权重大于负样本权重

        # mask[label == 2] = 0
        cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='none')
        return cost.mean([1, 2, 3]) #在c，h,w维度上求均值，返回每个batch的平均损失，形状为 (B, 1)
    
    def get_loss(self, pred, target, mean=True):
        self.loss_type = 'l2' #默认使用l2
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        #print("loss type:", self.loss_type, "loss:", loss)
        return loss


class SpecifyGradient(torch.autograd.Function): #负责将loss_tmp.mean()的梯度传递给x_rec,loss_tmp来自解码后的边缘图与真实边缘图的损失
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        # 前向传播函数，接收两个输入：
        # - input_tensor: 潜在表示 x_rec（即 \hat{z}_0），形状 (B, C, H, W)
        # - gt_grad: 目标梯度，通常是 \partial L_{wce} / \partial edge_rec，形状 (B,)
        # ctx: 上下文对象，用于保存反向传播所需的数据
        # @custom_fwd: 支持 AMP，确保前向传播兼容混合精度
        ctx.save_for_backward(gt_grad)
        # 保存目标梯度 gt_grad，以便在反向传播中使用
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
        # 返回一个虚拟值 1，形状与 input_tensor 相同，类型与 input_tensor 相同
        # 这是占位符输出，不影响前向计算，仅为满足 autograd 接口
        # 注释说明：返回全 1 是为了与 AMP 的 scaler 交互，在反向传播中获取缩放因子

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale): #反向传播函数，接收一个输入 grad_scale，表示梯度缩放因子（通常为 1，除非使用 AMP）。ctx: 上下文对象，包含保存的 gt_grad
        (gt_grad,) = ctx.saved_tensors   # 从 ctx 提取保存的 gt_grad，
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
    
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


