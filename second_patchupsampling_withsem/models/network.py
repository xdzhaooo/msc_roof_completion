import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from einops import rearrange,repeat
import torch.nn.functional as F

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
        
        gammas = np.cumprod(alphas, axis=0) #累积alpha，得到gamma列表
        gammas_prev = np.append(1., gammas[:-1]) #前一个gamma值，第一个gamma值为1，直到gamma_t-1

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas)) #注册为buffer，不会被优化器更新，但可通过self.gammas访问
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

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None, **kwargs): #p（xt|xt-1）#预测分布xt|xt-1
        
        padded_shape = kwargs.get('padded_shape', None) #padded_shape是padding后的图像大小，如果没有就用默认值
        patch_size = kwargs.get('patch_size', None) #patch_size是图像的patch大小，如果没有就用默认值
        halfp=int(patch_size[0]//2)
        ori_pos = kwargs.get('ori_pos', None) #ori_pos是原始图像的patch位置，如果没有就用默认值
        pad_pos = kwargs.get('pad_pos', None)
        #计算没有padding时batch_size
        _, c, h, w = padded_shape 
        b = y_t.shape[0] #batch_size

        p1_, p2_ = h//patch_size[0], w//patch_size[1]
        b = b // (p1_*p2_) # batch_size =
        # b = b *(p1_-1)*(p2_-1) 


        noise_level_ = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device) #噪声水平累积，从gamma中提取t时刻的噪声水平.输出形状为b,1
        rep = int(y_t.shape[0] // noise_level_.shape[0]) #重复次数，y_t的第一个维度是b，noise_level的第一个维度是b，这里是为了保证y_t和noise_level的维度一致
        noise_level = repeat(noise_level_, 'b 1 -> (b repeat) 1', repeat=rep) #将noise_level重复，形状为b p1 p2,1,这里是带padding的维度
        noise_level_crop_batch = repeat(noise_level_, 'b 1 -> (b repeat) 1', repeat=(p1_-1)*(p2_-1) )#这里是不带padding的patch的batch_size
        #print('noise_level:',noise_level.shape,'noise_level_crop_batch:',noise_level_crop_batch.shape,"noise_level_:",noise_level_.shape)   

        print('y_cond:',y_cond.shape)
        print('y_t:',y_t.shape)
        noise_dict=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), 
                                   noise_level,phase="not train",
                                   sample_gammas_ori_img_num=noise_level_crop_batch, 
                                   patch_size=patch_size,
                                   padded_shape=padded_shape,
                                   patch_pos = pad_pos,
                                   embed_patch_pos=ori_pos
                                   
                                   ) #将y_cond和y_t拼接起来，放入unet，得到带噪声输出
        noise=noise_dict['pred_cropped'] #得到的噪声b,p1,p2,c,h,w
        # print('noise:',noise.shape)
        assert noise.size(0) == (p1_-1)*(p2_-1)*b
        noise = rearrange(noise, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=p1_-1, p2=p2_-1)
        noise = F.pad(noise, (halfp,halfp,halfp,halfp), mode='constant', value=-1)
        noise = rearrange(noise, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h=patch_size[0], w=patch_size[1])
        t=repeat(t,'b -> (b repeat)',repeat=(p1_)*(p2_))

        y_0_hat = self.predict_start_from_noise(                             #预测初始值y0_hat
                y_t, t=t, noise=noise) #代入公式得到y0_hat
        

        if clip_denoised:   #将y0_hat的值限制在-1到1之间
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t) #预测后验分布yt-1|yt,y0_hat
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None,):    #正向采样，从y0采样到yt
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )
 
    @torch.no_grad()  #采样过程不需要梯度
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None, **kwargs):  #逆向采样，从yt采样到yt-1
        patch_size = kwargs.get('patch_size', None) #patch_size是图像的patch大小，如果没有就用默认值
        padded_shape = kwargs.get('padded_shape', None) #padded_shape是padding后的图像大小，如果没有就用默认值
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, patch_size=patch_size, padded_shape=padded_shape,kwargs=kwargs) #预测分布yt|yt-1
        t = repeat(t,'b -> (b repeat)',repeat=int(y_t.shape[0]//t.shape[0])) #重复t，使得t的维度和y_t的维度一致
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t) #随机噪声
        return model_mean + noise * (0.5 * model_log_variance).exp()   #mean + noise * std. 因为model_log_variance实际上是log(std^2)，这里乘以noise的是std

    @torch.no_grad()
    def p_sample_ddim(self, y_t, t,t_prev, clip_denoised=True, y_cond=None, eta=0.0,patch_size=None,padded_shape=None,ori_img_size=None,ori_pos=None,pad_pos=None,first_output=None):  # 使用ddim进行逆向采样
        #y_t, t_tensor,t_tensor_prev, y_cond=y_cond,patch_size=patch_size,padded_shape=padded_shape, ori_img_size=ori_img_size, ori_pos=ori_pos, pad_pos=pad_pos
        # 1. 预测 y₀_hat（初始图像估计）
        # noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # y0_hat = self.predict_start_from_noise(
        #     y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
        # )

        # if clip_denoised:
        #     y0_hat.clamp_(-1., 1.)
        halfp=int(patch_size[0]//2)
        #计算没有padding时batch_size
        _, c, h, w = padded_shape 
        b = y_t.shape[0] #batch_size

        p1_, p2_ = h//patch_size[0], w//patch_size[1]
        b = b // (p1_*p2_) # batch_size =
        # b = b *(p1_-1)*(p2_-1) 


        noise_level_ = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device) #噪声水平累积，从gamma中提取t时刻的噪声水平.输出形状为b,1
        rep = int(y_t.shape[0] // noise_level_.shape[0]) #重复次数，y_t的第一个维度是b，noise_level的第一个维度是b，这里是为了保证y_t和noise_level的维度一致
        noise_level = repeat(noise_level_, 'b 1 -> (b repeat) 1', repeat=rep) #将noise_level重复，形状为b p1 p2,1,这里是带padding的维度
        noise_level_crop_batch = repeat(noise_level_, 'b 1 -> (b repeat) 1', repeat=(p1_-1)*(p2_-1) )#这里是不带padding的patch的batch_size
        #print('noise_level:',noise_level.shape,'noise_level_crop_batch:',noise_level_crop_batch.shape,"noise_level_:",noise_level_.shape)   

        noise_dict=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), 
                                   noise_level,
                                   phase="not train",
                                   sample_gammas_ori_img_num=noise_level_crop_batch, 
                                   patch_size=patch_size,
                                   padded_shape=padded_shape,
                                   patch_pos = pad_pos,
                                   embed_patch_pos=ori_pos,
                                   first_output=first_output
                                   ) #将y_cond和y_t拼接起来，放入unet，得到带噪声输出
        noise=noise_dict['pred_cropped'] #得到的噪声b,p1,p2,c,h,w
        # print('noise:',noise.shape)
        assert noise.size(0) == (p1_-1)*(p2_-1)*b
        noise = rearrange(noise, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=p1_-1, p2=p2_-1)
        noise = F.pad(noise, (halfp,halfp,halfp,halfp), mode='constant', value=-1)
        noise = rearrange(noise, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h=patch_size[0], w=patch_size[1])
        t=repeat(t,'b -> (b repeat)',repeat=(p1_)*(p2_))
        t_prev=repeat(t_prev,'b -> (b repeat)',repeat=(p1_)*(p2_))

        y0_hat = self.predict_start_from_noise(                             #预测初始值y0_hat
                y_t, t=t, noise=noise) #代入公式得到y0_hat
        

        if clip_denoised:   #将y0_hat的值限制在-1到1之间
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
            return y0_hat

        # 4. 计算DDIM中的σ_t, 控制随机性
        # print('t:',t.shape,'t_prev:',t_prev.shape)
        # print('alpha_prev:',alpha_prev.shape,'alpha_t:',alpha_t.shape)
        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)

        # 5. DDIM更新公式：由确定性部分和随机部分构成
        y_t_prev = (torch.sqrt(alpha_prev) * y0_hat +
                    torch.sqrt(1 - alpha_prev - sigma_t**2) * epsilon_pred)
        if eta > 0:
            y_t_prev = y_t_prev + sigma_t * torch.randn_like(y_t)

        return y_t_prev
    
    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8,ddim_sample_steps=50,**kwargs): #恢复图像的完整过程
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

        ori_img_size = kwargs.get('ori_img_size', None) #原始图像的大小，未经过padding，如果没有就用默认值
        patch_size = kwargs.get('patch_size', None) #patch_size是图像的patch大小，如果没有就用默认值
        halfp=int(patch_size[0]//2)
        if len(patch_size)==1:
            p_size = patch_size[0].item()
            patch_size=[p_size,p_size]
        
        first_output = kwargs.get('first_output', None) #第一次diffusion的输出结果，如果没有就用默认值

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
                #y_t = default(y_t, lambda: torch.randn_like(y_cond)) #如果yt不存在，就用随机噪声代替，大多数情况下是这样，但是
        y_t_shape=y_cond[:,:1,:,:]
        # print('y_t_shape:',y_t_shape.shape) 
        #y_t = default(y_t[:,:1,:,:], lambda: torch.randn_like(y_t_shape)) #如果yt不存在，就用随机噪声代替，大多数情况下是这样，但是不知道这里为什么val阶段用cond作为yt
        
        y_t = default(None, lambda: torch.randn_like(y_t_shape))
        ret_arr = [y_t] 
        # print('y_t:',y_t.shape)
        ori_patch_num = [y_cond.shape[2]//patch_size[0], y_cond.shape[3]//patch_size[1]] #原始图像的patch数量
        # 确保 ori_patch_num 是整数，防止 Tensor 类型错误
        ori_patch_num = [int(ori_patch_num[0]), int(ori_patch_num[1])]
        rows, cols = torch.meshgrid(torch.arange(ori_patch_num[0]), torch.arange(ori_patch_num[1]), indexing='ij')
        # 修正：原始图像在填充图像中从(halfp, halfp)开始，需要加上偏移量
        rows = (rows * patch_size[0] + patch_size[0]//2) / (y_cond.shape[2]+patch_size[0]) #原始图像的patch在填充图像中的左上角位置（考虑padding偏移）
        cols = (cols * patch_size[1] + patch_size[1]//2) / (y_cond.shape[3]+patch_size[1])
        ori_pos = torch.stack([rows, cols], dim=-1) #原始图像的patch在填充图像中的左上角位置，形状为p1*p2,2
        ori_pos = repeat(ori_pos,'c h w -> b c h w',b=b) #将原始图像的patch位置重复，形状为b,p1,p2,2
        ori_pos = rearrange(ori_pos,'b h w n -> (b h w) n') #将原始图像的patch位置的维度从b,h,w,n变为b*h*w,n

        
        rows,cols = torch.meshgrid(torch.arange(ori_patch_num[0]+1), torch.arange(ori_patch_num[1]+1), indexing='ij')
        rows = rows * (patch_size[0]/(y_cond.shape[2]+patch_size[0])) #填充图像的patch左上角位置，形状为p1*p2
        cols = cols * (patch_size[1]/(y_cond.shape[3]+patch_size[1]))
        pad_pos = torch.stack([rows, cols], dim=-1) #填充图像的patch左上角位置，形状为p1*p2,2
        pad_pos = repeat(pad_pos,'c h w -> b c h w',b=b) #将原始图像的patch位置重复，形状为b,p1,p2,2
        pad_pos = rearrange(pad_pos,'b h w n -> (b h w) n') #将原始图像的patch位置的维度从b,h,w,n变为b*h*w,n


        y_cond = F.pad(y_cond, (halfp,halfp,halfp,halfp), mode='constant', value=-1) #y_cond进行padding,说明y_cond的形状为原始图片
        padded_shape = y_cond.shape
        # print('padded_shape:',padded_shape) 
        #print("y_cond.shape:",y_cond.shape,"patch_size:",patch_size)
        y_cond = rearrange(y_cond,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将y_cond的维度从（b,c,p1*h,p2*w）变为（b*p1*p2,c,h,w）



        for t_val,t_prev in zip(timesteps, timesteps_prev):
            y_t = F.pad(y_t, (halfp,halfp,halfp,halfp), mode='constant', value=-1)
           
            y_t = rearrange(y_t,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将yt的维度从（b,c,p1*h,p2*w）变为（b*p1*p2,c,h,w）

            t_tensor = torch.full((b,), t_val.item(), device=y_cond.device, dtype=torch.long)
            t_tensor_prev = torch.full((b,), t_prev.item(), device=y_cond.device, dtype=torch.long)
            # 使用 DDIM 采样函数 p_sample_ddim 进行采样
            
            y_t = self.p_sample_ddim(y_t, t_tensor,t_tensor_prev, y_cond=y_cond,patch_size=patch_size,padded_shape=padded_shape, ori_img_size=ori_img_size, ori_pos=ori_pos, pad_pos=pad_pos,first_output=first_output)#逆向采样，从yt采样到yt-1

            y_t_new = rearrange(y_t,'(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=padded_shape[2]//patch_size[0], p2=padded_shape[3]//patch_size[1]) #将yt的维度从（b*p1*p2,c,h,w）变为（b,c,p1*h,p2*w）
            y_t = y_t_new[:, :, halfp:-halfp, halfp:-halfp] #裁剪掉边缘，维度为b,c,h,w
            # 如果是补全任务，保留已知区域
            if mask is not None and y_0 is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            ret_arr.append(y_t)
        
        # 拼接所有采样的中间结果
        ret_arr = torch.cat(ret_arr, dim=0)
        return y_t, ret_arr
    

    
    # @torch.no_grad()
    # def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, **kwargs): #恢复图像的完整过程 sample_num是去噪过程中抽取这些次保留图片，方便后续评估
    #     b, *_ = y_cond.shape
    #     #y_cond是已知的部分或者叫做条件部分，y_t是要采样的部分，y_0是原始图像（如果有），mask是已知的部分（如果有）
    #     assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
    #     sample_inter = None if sample_num == 0 else (self.num_timesteps//sample_num) #采样间隔，每隔多少时间步评估一次
    #     ori_img_size = kwargs.get('ori_img_size', None) #原始图像的大小，未经过padding，如果没有就用默认值

    #     patch_size = kwargs.get('patch_size', None) #patch_size是图像的patch大小，如果没有就用默认值
    #     halfp=int(patch_size[0]//2)
        
        
    #     #y_t = default(y_t, lambda: torch.randn_like(y_cond)) #如果yt不存在，就用随机噪声代替，大多数情况下是这样，但是
    #     y_t_shape=y_cond[:,:1,:,:]
    #     # print('y_t_shape:',y_t_shape.shape) 
    #     #y_t = default(y_t[:,:1,:,:], lambda: torch.randn_like(y_t_shape)) #如果yt不存在，就用随机噪声代替，大多数情况下是这样，但是不知道这里为什么val阶段用cond作为yt
    #     y_t = default(None, lambda: torch.randn_like(y_t_shape))
    #     ret_arr = y_t 
    #     # print('y_t:',y_t.shape)

    #     ori_patch_num = [y_cond.shape[2]//patch_size[0], y_cond.shape[3]//patch_size[1]] #原始图像的patch数量
    #     rows, cols = torch.meshgrid(torch.arange(ori_patch_num[0]), torch.arange(ori_patch_num[1]), indexing='ij')
    #     rows = (rows+0.5) * patch_size[0]/y_cond.shape[2] #原始图像的patch位置，形状为p1*p2
    #     cols = (cols+0.5) * patch_size[1]/y_cond.shape[3]
    #     ori_pos = torch.stack([rows, cols], dim=-1) #原始图像的patch位置，形状为p1*p2,2
    #     ori_pos = repeat(ori_pos,'c h w -> b c h w',b=b) #将原始图像的patch位置重复，形状为b,p1,p2,2
    #     ori_pos = rearrange(ori_pos,'b h w n -> (b h w) n') #将原始图像的patch位置的维度从b,h,w,n变为b*h*w,n

        
    #     rows,cols = torch.meshgrid(torch.arange(ori_patch_num[0]+1), torch.arange(ori_patch_num[1]+1), indexing='ij')
    #     rows = (rows+0.5) * (patch_size[0]/y_cond.shape[2]+1) #原始图像的patch位置，形状为p1*p2
    #     cols = (cols+0.5) * (patch_size[1]/y_cond.shape[3]+1)
    #     pad_pos = torch.stack([rows, cols], dim=-1) #原始图像的patch位置，形状为p1*p2,2
    #     pad_pos = repeat(pad_pos,'c h w -> b c h w',b=b) #将原始图像的patch位置重复，形状为b,p1,p2,2
    #     pad_pos = rearrange(pad_pos,'b h w n -> (b h w) n') #将原始图像的patch位置的维度从b,h,w,n变为b*h*w,n


    #     y_cond = F.pad(y_cond, (halfp,halfp,halfp,halfp), mode='constant', value=-1) #y_cond进行padding,说明y_cond的形状为原始图片
    #     padded_shape = y_cond.shape
    #     # print('padded_shape:',padded_shape) 
    #     y_cond = rearrange(y_cond,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将y_cond的维度从（b,c,p1*h,p2*w）变为（b*p1*p2,c,h,w）


    #     #采样到t=0
    #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):

    #         y_t = F.pad(y_t, (halfp,halfp,halfp,halfp), mode='constant', value=-1)
           
    #         y_t = rearrange(y_t,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将yt的维度从（b,c,p1*h,p2*w）变为（b*p1*p2,c,h,w）

    #         t = torch.full((b,), i, device=y_cond.device, dtype=torch.long) #形状为b的全为i的张量

    #         y_t = self.p_sample(y_t, t, y_cond=y_cond,patch_size=patch_size,padded_shape=padded_shape, ori_img_size=ori_img_size, ori_pos=ori_pos, pad_pos=pad_pos)#逆向采样，从yt采样到yt-1

    #         y_t_new = rearrange(y_t,'(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=padded_shape[2]//patch_size[0], p2=padded_shape[3]//patch_size[1]) #将yt的维度从（b*p1*p2,c,h,w）变为（b,c,p1*h,p2*w）
    #         y_t = y_t_new[:, :, halfp:-halfp, halfp:-halfp] #裁剪掉边缘，维度为b,c,h,w

    #         if mask is not None: #在补全任务中，mask是已知的，所以需要将已知的部分保留下来
    #             y_t = y_0*(1.-mask) + mask*y_t#y0是图像补全任务中的原始图像，mask是已知的部分，y_t是采样的图像（roofdiffusion里，mask就是footprint）
    #         if sample_inter is not None and i % sample_inter == 0:
    #             ret_arr = torch.cat([ret_arr, y_t], dim=0) #将采样的图像拼接起来，方便后续的评估
    #     return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None, **kwargs): #训练过程
        loss_mask = kwargs.get('loss_mask', None) #loss_mask是用于计算损失的mask，如果没有就用mask
        patch_size = kwargs.get('patch_size', None) #patch_size是图像的patch大小，如果没有就用默认值
        halfp=int(patch_size[0]//2)
        patch_pos = kwargs.get('patch_pos', None) #patch_pos是patch的位置，如果没有就用默认值
        ori_img_size = kwargs.get('ori_img_size', None) #原始完整图像的大小，未经过padding，如果没有就用默认值
        first_output = kwargs.get('first_output', None) #第一次diffusion的输出结果，如果没有就用默认值

        mask_crop = mask[:, :, halfp:-halfp, halfp:-halfp] #裁剪掉边缘的mask，形状为b，c，h，w

        # sampling from p(gammas)
        b, *_ = y_0.shape #b是batch_size,y0的形状为b，c，p1 h，p2 w
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() #随机采样一个时间步，形状为b，。
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1)) #提取t-1时刻的gamma，形状为b,-》b,1,1
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))    #提取t时刻的gamma，形状为b,-》b,1,1
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1  
        #这一步的目的是在两个连续时间步之间随机采样一个值，使得训练中使用的 gamma 参数有一定的连续性和随机性。
        sample_gammas = sample_gammas.view(b, -1)#将采样的gamma值reshape成b,1

        noise = default(noise, lambda: torch.randn_like(y_0)) #如果噪声不存在，就用随机噪声代替。噪声的形状和y0一样. y0形状为b，c，p1 h，p2 w. 这是padding后的维度
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise) #加噪后的图像,形状为b，1，h，w
        
        #patching 有用
        rep = int(b/sample_gammas.shape[0])
        sample_gammas_ori_img_num = repeat(sample_gammas,'b 1 -> (b repeat) 1',repeat=rep) #将采样的gamma值重复，形状为b p1 p2,1,这里是没有padding时的维度
        #print('sample_gammas_ori_img_num:',sample_gammas_ori_img_num.shape,'sample_gammas:',sample_gammas.shape)

        y_noisy=rearrange(y_noisy,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将加噪后的图像切成patch
        noise = rearrange(noise,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将噪声切成patch
        loss_mask = rearrange(loss_mask,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将mask切成patch
        sample_gammas = repeat(sample_gammas,'b 1 -> (b repeat) 1',repeat=y_noisy.shape[0]//y_0.shape[0]) #将采样的gamma值重复，形状为b p1 p2,1
        y_cond = rearrange(y_cond,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将条件部分切成patch
        y_0 = rearrange(y_0,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1]) #将原始图像切成patch
        patch_pos_padded = rearrange(patch_pos,'a b c d -> (a b c) d') #将patch位置切成patch，形状为b*2*2,2
        center = patch_pos.mean(dim=(1, 2)) #计算patch位置的中心，形状为b,2



        if mask is not None:
            mask = rearrange(mask,'b c (p1 h) (p2 w) -> (b p1 p2) c h w',h=patch_size[0], w=patch_size[1])
        
        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), 
                                        gammas=sample_gammas,
                                        loss_mask=loss_mask,
                                        patch_size=patch_size,
                                        patch_pos=patch_pos_padded,
                                        ori_img_size=ori_img_size,
                                        sample_gammas_ori_img_num=sample_gammas_ori_img_num,
                                        embed_patch_pos=center,
                                        first_output=first_output) 
            model_output_no_shift = noise_hat['pred_all']
            model_output_shift = noise_hat['pred_cropped']
            assert model_output_no_shift.size(0) == noise.size(0) #改后
            noise_ori = rearrange(noise, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = 2, p2 = 2) #将noise的维度从（2*2*b,c,patch_size,pacth_size）变为（b,c,2*patch_size,2*patch_size）
            noise_ori_crop = noise_ori[:, :, halfp:-halfp, halfp:-halfp] #对noise_ori进行裁剪，裁剪掉边缘，维度为（b,c,patch_size,patch_size）
            noise_target_shift = rearrange(noise_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size[0], w = patch_size[1]) #将noise_ori_crop的维度从（b,c,patch_size,patch_size）变为（b*p1*p2,c,h,w），p1和p2都为1，h和w都为patch_size

            assert model_output_no_shift.size() == noise.size() #改后
            assert model_output_shift.size() == noise_target_shift.size() #改后


            #mask存在时，只保留mask部分，当前设置mask就是footprint
            #这里concate了y_cond和y_noisy，维度为p1p2b，2c，h，w
            # from torchinfo import summary
            # params = {}#unet.copy()
            # params['input_size'] = (4,2,256,256)
            # params['gammas'] = sample_gammas
            # summary(self.denoise_fn, **params, col_names=("input_size", "output_size", "num_params"), depth=10)

            if hasattr(self.loss_fn, '__name__') and self.loss_fn.__name__ in ['masked_mse_loss', 'masked_l1_loss']:
                loss_no_shift = self.loss_fn(mask*noise, mask*model_output_no_shift, mask) # might not be necessary 
                loss_shift = self.loss_fn(mask_crop*noise_ori_crop, mask_crop*model_output_shift, mask_crop) # might not be necessary
            else:
                # loss = self.loss_fn(mask*noise, mask*noise_hat)
                loss_no_shift = self.loss_fn(mask*noise, mask*model_output_no_shift)
                loss_shift = self.loss_fn(mask_crop*noise_ori_crop, mask_crop*model_output_shift)
        else:
            # noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas) #从y_cond和y_noisy中去噪声，通过unet
            #mask不存在时，直接去噪声，不需要保留mask部分。也就是说此时的y_noisy在mask外的部分也有噪声
            # loss = self.loss_fn(noise, noise_hat)
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1),
                                        gammas=sample_gammas,
                                        loss_mask=loss_mask,
                                        patch_size=patch_size,
                                        patch_pos=patch_pos,
                                        ori_img_size=ori_img_size,
                                        sample_gammas_ori_img_num=sample_gammas_ori_img_num)
            model_output_no_shift = noise_hat['pred_all']
            model_output_shift = noise_hat['pred_cropped']
            loss_no_shift = self.loss_fn(noise, model_output_no_shift)
            loss_shift = self.loss_fn(noise, model_output_shift)
        return loss_no_shift+loss_shift


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
    out = a.gather(-1, t) #在a的最后一个维度上，根据t的索引值，将a中的值取出来。与t形状相同。形状为b,1
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


