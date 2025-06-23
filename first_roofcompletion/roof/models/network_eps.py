import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet) #解包unet參數，UNet(in_channels=3, out_channels=3, num_layers=4, ...)，用config中的参数初始化unet
        self.beta_schedule = beta_schedule #beta调度

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
    def p_sample_ddim(self, y_t, t,t_prev, clip_denoised=True, y_cond=None, eta=0.0,control_image=None):  # 使用ddim进行逆向采样
        # 1. 预测 y₀_hat（初始图像估计）
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level,control_image)
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
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        # y_cond_mask = (y_cond > -1. ).to(y_cond.device).float() #将y_cond中大于-1的部分转换为1，小于-1的部分转换为0
        # y_t_with_cond = y_t * (1. - y_cond_mask) + y_cond * y_cond_mask #将条件部分加入到y_t中
        # y_t = y_t_with_cond 

        t_tensor = torch.full((b,), self.num_timesteps - 1, device=y_t.device, dtype=torch.long)
        gamma_t1 = extract(self.gammas, t_tensor, x_shape=(1, 1)) #提取t-1时刻的gamma，形状为b,-》b,1,1
        y_t=self.q_sample(y_0=y_cond, sample_gammas=gamma_t1.view(-1, 1, 1, 1), noise=torch.randn(y_cond.shape).to(y_cond.device)) #加噪后的图像

        control_image = torch.zeros_like(y_cond)

        ret_arr = [y_t]
        ret_y0 = [torch.zeros_like(y_t)]
        # 采样过程

        for t_val,t_prev in zip(timesteps, timesteps_prev):
            t_tensor = torch.full((b,), t_val.item(), device=y_cond.device, dtype=torch.long)
            t_tensor_prev = torch.full((b,), t_prev.item(), device=y_cond.device, dtype=torch.long)
            # 使用 DDIM 采样函数 p_sample_ddim 进行采样
            y_t,y0_hat = self.p_sample_ddim(y_t, t_tensor,t_tensor_prev, y_cond=y_cond,control_image=control_image)
            # 如果是补全任务，保留已知区域
            if mask is not None and y_0 is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            ret_arr.append(y_t)
            ret_y0.append(y0_hat)
        
        # 拼接所有采样的中间结果, ret arr will be saved as a tensor
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
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise) #加噪后的图像

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,control_image)
            #mask存在时，只保留mask部分
            # from torchinfo import summary
            # params = {}#unet.copy()
            # params['input_size'] = (4,2,256,256)
            # params['gammas'] = sample_gammas
            # summary(self.denoise_fn, **params, col_names=("input_size", "output_size", "num_params"), depth=10)

            if hasattr(self.loss_fn, '__name__') and self.loss_fn.__name__ in ['masked_mse_loss', 'masked_l1_loss', 'combined_loss_with_masked_ssim']:
                loss = self.loss_fn(mask*noise, mask*noise_hat, mask) # might not be necessary 
            else:
                loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas) #从y_cond和y_noisy中去噪声，通过unet
            #mask不存在时，直接去噪声，不需要保留mask部分。也就是说此时的y_noisy在mask外的部分也有噪声
            loss = self.loss_fn(noise, noise_hat)
        return loss


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


