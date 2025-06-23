import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import os
import sys
import json
from core.base_network import BaseNetwork

class Network(BaseNetwork):
    def __init__(self, unetMom, beta_schedule, module_name='mom_modules', **kwargs):
        super(Network, self).__init__(**kwargs)
        
        # 根据module_name导入对应的UNet
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        elif module_name == 'mom_modules':
            from .mom_moduls.unet import UNet
        else:
            # 默认使用mom_modules
            from .mom_moduls.unet import UNet
        
        # 只初始化MOM模块
        self.MOM = UNet(**unetMom)
        self.beta_schedule = beta_schedule
        
        # 预训练模型将在load_pretrained_models方法中初始化
        self.Roofdenoise_fn = None
        self.RoofLinedenoise_fn = None
        
        # 添加参数化方式，默认为噪声预测
        self.parameterization = kwargs.get('parameterization', 'eps')
        self.parameterization='v'

    def load_pretrained_models(self, roof_path, roofline_path):
        """
        加载预训练的roof和roofline模型
        
        Args:
            roof_path (str): roof模型的路径
            roofline_path (str): roofline模型的路径
        """
        print("="*60)
        print("开始加载预训练模型...")
        print("="*60)
        
        roof_success = False
        roofline_success = False
        
        # 加载roof模型
        if roof_path and os.path.exists(roof_path):
            print(f"正在加载Roof模型: {roof_path}")
            roof_success = self._load_roof_model(roof_path)
        else:
            print(f"❌ Roof模型路径不存在: {roof_path}")
            
        # 加载roofline模型  
        if roofline_path and os.path.exists(roofline_path):
            print(f"正在加载Roofline模型: {roofline_path}")
            roofline_success = self._load_roofline_model(roofline_path)
        else:
            print(f"❌ Roofline模型路径不存在: {roofline_path}")
        
        # 将预训练模型移动到GPU（如果CUDA可用）
        if torch.cuda.is_available():
            if hasattr(self, 'Roofdenoise_fn') and self.Roofdenoise_fn is not None:
                self.Roofdenoise_fn = self.Roofdenoise_fn.cuda()
                print(f"✅ Roof模型已移动到GPU: {next(self.Roofdenoise_fn.parameters()).device}")
            
            if hasattr(self, 'RoofLinedenoise_fn') and self.RoofLinedenoise_fn is not None:
                self.RoofLinedenoise_fn = self.RoofLinedenoise_fn.cuda()
                print(f"✅ Roofline模型已移动到GPU: {next(self.RoofLinedenoise_fn.parameters()).device}")
        
        # 检查加载结果并断言
        print("\n" + "="*60)
        print("预训练模型加载结果检查")
        print("="*60)
        print(f"Roof模型加载: {'✅ 成功' if roof_success else '❌ 失败'}")
        print(f"Roofline模型加载: {'✅ 成功' if roofline_success else '❌ 失败'}")
        
        # 断言检查 - 如果任何一个模型加载失败，程序退出
        assert roof_success, f"❌ Roof模型加载失败！请检查模型路径和配置: {roof_path}"
        assert roofline_success, f"❌ Roofline模型加载失败！请检查模型路径和配置: {roofline_path}"
        
        # 验证模型是否正确加载到内存
        assert self.Roofdenoise_fn is not None, "❌ Roof模型未正确初始化到self.Roofdenoise_fn"
        assert self.RoofLinedenoise_fn is not None, "❌ Roofline模型未正确初始化到self.RoofLinedenoise_fn"
        
        print("🎉 所有预训练模型加载成功！")
        print("="*60)
        
        # 打印参数统计
        self._print_trainable_parameters()
    
    def _load_roof_model(self, roof_path):
        """加载roof模型"""
        try:
            # 获取roof目录路径
            roof_dir = os.path.dirname(roof_path)
            config_path = os.path.join(roof_dir, 'config.json')
            
            if not os.path.exists(config_path):
                print(f"❌ Roof配置文件不存在: {config_path}")
                return False
                
            # 读取配置文件
            with open(config_path, 'r') as f:
                roof_config = json.load(f)
            
            # 获取网络配置
            network_config = roof_config['model']['which_networks'][0]['args']
            unet_config = network_config['unet']
            module_name = network_config.get('module_name', 'roof_modules')
            
            # 保存当前的sys.path状态
            original_path = sys.path.copy()
            
            try:
                # 将roof相关路径添加到最前面
                roof_models_path = os.path.join(roof_dir, 'models')
                sys.path.insert(0, roof_models_path)
                sys.path.insert(0, roof_dir)
                
                print(f"🔧 设置导入路径: {roof_models_path}")
                
                # 使用传统导入方式
                if module_name == 'roof_modules':
                    from roof_modules.unet import UNet as RoofUNet
                elif module_name == 'guided_diffusion_modules':
                    from guided_diffusion_modules.unet import UNet as RoofUNet
                else:
                    # 默认尝试roof_modules
                    from roof_modules.unet import UNet as RoofUNet
                    
                print(f"✅ 成功导入UNet模块: {module_name}")
                
            finally:
                # 恢复原始的sys.path
                sys.path = original_path
            
            # 创建roof模型
            self.Roofdenoise_fn = RoofUNet(**unet_config)
            print(f"✅ 成功创建Roof模型")
            
            # 加载预训练权重
            checkpoint = torch.load(roof_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # 过滤权重
            filtered_dict = {}
            for k, v in state_dict.items():
                if k.startswith('denoise_fn.'):
                    new_key = k.replace('denoise_fn.', '')
                    filtered_dict[new_key] = v
                elif 'denoise_fn' not in k:
                    filtered_dict[k] = v
                    
            # 加载权重，使用strict=False忽略不匹配的参数
            missing_keys, unexpected_keys = self.Roofdenoise_fn.load_state_dict(filtered_dict, strict=False)
            
            # 计算加载成功率
            total_params = len(self.Roofdenoise_fn.state_dict())
            loaded_params = total_params - len(missing_keys)
            success_rate = loaded_params / total_params if total_params > 0 else 0
            
            print(f"✅ Roof权重加载完成: {loaded_params}/{total_params} 参数 ({success_rate*100:.1f}%)")
            if missing_keys:
                print(f"⚠️  缺失参数: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"⚠️  多余参数: {len(unexpected_keys)} 个")
            
            # 冻结参数，但保持init_cond_net和cond_proj_in可训练
            for name, param in self.Roofdenoise_fn.named_parameters():
                if 'init_cond_net' in name or 'cond_proj_in' in name:
                    param.requires_grad = True
                    print(f"✅ 保持可训练: {name}")
                else:
                    param.requires_grad = False
                
            print(f"✅ Roof模型加载成功: {roof_path}")
            return True
            
        except Exception as e:
            print(f"❌ Roof模型加载失败: {e}")
            print("详细错误信息:")
            import traceback
            traceback.print_exc()
            return False

    def _load_roofline_model(self, roofline_path):
        """加载roofline模型"""
        try:
            # 获取roofline目录路径
            roofline_dir = os.path.dirname(roofline_path)
            config_path = os.path.join(roofline_dir, 'config.json')
            
            if not os.path.exists(config_path):
                print(f"❌ Roofline配置文件不存在: {config_path}")
                return False
                
            # 读取配置文件
            with open(config_path, 'r') as f:
                roofline_config = json.load(f)
            
            # 获取网络配置
            network_config = roofline_config['model']['which_networks'][0]['args']
            unet_config = network_config['unet']
            module_name = network_config.get('module_name', 'roofline_modules')
            
            # 保存当前的sys.path状态
            original_path = sys.path.copy()
            
            try:
                # 将roofline相关路径添加到最前面
                roofline_models_path = os.path.join(roofline_dir, 'models')
                sys.path.insert(0, roofline_models_path)
                sys.path.insert(0, roofline_dir)
                
                print(f"🔧 设置导入路径: {roofline_models_path}")
                
                # 使用传统导入方式
                if module_name == 'roofline_modules':
                    from roofline_modules.unet import UNet as RooflineUNet
                elif module_name == 'guided_diffusion_modules':
                    from guided_diffusion_modules.unet import UNet as RooflineUNet
                else:
                    # 默认尝试roofline_modules
                    from roofline_modules.unet import UNet as RooflineUNet
                    
                print(f"✅ 成功导入UNet模块: {module_name}")
                
            finally:
                # 恢复原始的sys.path
                sys.path = original_path
            
            # 创建roofline模型
            self.RoofLinedenoise_fn = RooflineUNet(**unet_config)
            print(f"✅ 成功创建Roofline模型")
            
            # 加载预训练权重
            checkpoint = torch.load(roofline_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # 过滤权重
            filtered_dict = {}
            for k, v in state_dict.items():
                if k.startswith('denoise_fn.'):
                    new_key = k.replace('denoise_fn.', '')
                    filtered_dict[new_key] = v
                elif 'denoise_fn' not in k:
                    filtered_dict[k] = v
                    
            # 加载权重，使用strict=False忽略不匹配的参数
            missing_keys, unexpected_keys = self.RoofLinedenoise_fn.load_state_dict(filtered_dict, strict=False)
            
            # 计算加载成功率
            total_params = len(self.RoofLinedenoise_fn.state_dict())
            loaded_params = total_params - len(missing_keys)
            success_rate = loaded_params / total_params if total_params > 0 else 0
            
            print(f"✅ Roofline权重加载完成: {loaded_params}/{total_params} 参数 ({success_rate*100:.1f}%)")
            if missing_keys:
                print(f"⚠️  缺失参数: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"⚠️  多余参数: {len(unexpected_keys)} 个")
            

            for name, param in self.RoofLinedenoise_fn.named_parameters():
                if 'context_compressor' in name:
                    param.requires_grad = True
                    print(f"✅ 保持可训练: {name}")
                else:
                    param.requires_grad = False
                
            print(f"✅ Roofline模型加载成功: {roofline_path}")
            return True
            
        except Exception as e:
            print(f"❌ Roofline模型加载失败: {e}")
            print("详细错误信息:")
            import traceback
            traceback.print_exc()
            return False

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

    def _print_trainable_parameters(self):
        """打印各组件的可训练参数统计"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        def count_total_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        roof_trainable = count_parameters(self.Roofdenoise_fn)
        roof_total = count_total_parameters(self.Roofdenoise_fn)
        
        roofline_trainable = count_parameters(self.RoofLinedenoise_fn)
        roofline_total = count_total_parameters(self.RoofLinedenoise_fn)
        
        mom_trainable = count_parameters(self.MOM)
        mom_total = count_total_parameters(self.MOM)
        
        total_trainable = roof_trainable + roofline_trainable + mom_trainable
        total_params = roof_total + roofline_total + mom_total
        
        print("\n" + "="*60)
        print("MODEL PARAMETER STATISTICS")
        print("="*60)
        print(f"Roof Denoiser:     {roof_trainable:,} / {roof_total:,} trainable")
        print(f"Roofline Denoiser: {roofline_trainable:,} / {roofline_total:,} trainable")
        print(f"MOM:               {mom_trainable:,} / {mom_total:,} trainable")
        print("-"*60)
        print(f"Total:             {total_trainable:,} / {total_params:,} trainable")
        print(f"Trainable ratio:   {total_trainable/total_params*100:.2f}%")
        print("="*60)
    
    def unfreeze_component(self, component_name):
        """
        解冻指定组件
        
        Args:
            component_name (str): 'roof', 'roofline', 或 'mom'
        """
        if component_name.lower() == 'roof':
            for param in self.Roofdenoise_fn.parameters():
                param.requires_grad = True
            print("Roof denoiser unfrozen!")
        elif component_name.lower() == 'roofline':
            for param in self.RoofLinedenoise_fn.parameters():
                param.requires_grad = True
            print("Roofline denoiser unfrozen!")
        elif component_name.lower() == 'mom':
            for param in self.MOM.parameters():
                param.requires_grad = True
            print("MOM unfrozen!")
        else:
            print(f"Unknown component: {component_name}")
            
        self._print_trainable_parameters()
    
    def freeze_component(self, component_name):
        """
        冻结指定组件
        
        Args:
            component_name (str): 'roof', 'roofline', 或 'mom'
        """
        if component_name.lower() == 'roof':
            for name, param in self.Roofdenoise_fn.named_parameters():
                if 'init_cond_net' in name or 'cond_proj_in' in name:
                    param.requires_grad = True
                    print(f"✅ 保持可训练: {name}")
                else:
                    param.requires_grad = False
            print("Roof denoiser frozen (except init_cond_net and cond_proj_in)!")
        elif component_name.lower() == 'roofline':
            for param in self.RoofLinedenoise_fn.parameters():
                param.requires_grad = False
            print("Roofline denoiser frozen!")
        elif component_name.lower() == 'mom':
            for param in self.MOM.parameters():
                param.requires_grad = False
            print("MOM frozen!")
        else:
            print(f"Unknown component: {component_name}")
            
        self._print_trainable_parameters()

    def freeze_component_completely(self, component_name):
        """
        完全冻结指定组件（包括init_cond_net和cond_proj_in）
        
        Args:
            component_name (str): 'roof', 'roofline', 或 'mom'
        """
        if component_name.lower() == 'roof':
            for param in self.Roofdenoise_fn.parameters():
                param.requires_grad = False
            print("Roof denoiser completely frozen!")
        elif component_name.lower() == 'roofline':
            for param in self.RoofLinedenoise_fn.parameters():
                param.requires_grad = False
            print("Roofline denoiser completely frozen!")
        elif component_name.lower() == 'mom':
            for param in self.MOM.parameters():
                param.requires_grad = False
            print("MOM completely frozen!")
        else:
            print(f"Unknown component: {component_name}")
            
        self._print_trainable_parameters()

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
    def p_sample_ddim_rl(self, y_t, t,t_prev, clip_denoised=True, y_cond=None, eta=0.2,control_image=None, self_cond=None): #DDIM采样，从yt采样到yt-1
        # 1. 预测 y₀_hat（初始图像估计）
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # 使用RoofLinedenoise_fn进行roofline采样
        model_output = self.RoofLinedenoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level, control_image, self_cond=self_cond)

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
    
    @torch.no_grad()
    def p_sample_ddim_r(self, y_t, t,t_prev, clip_denoised=True, y_cond=None, eta=0.0,control_image=None):  # 使用ddim进行逆向采样
        # 1. 预测 y₀_hat（初始图像估计）
        # noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # y0_hat = self.predict_start_from_noise(
        #     y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level,control_image)
        # )

        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # 使用Roofdenoise_fn进行roof采样
        model_output = self.Roofdenoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level, control_image)

        if self.parameterization == "v":
            # Model predicts v, compute x₀_hat from v
            y0_hat = self.predict_start_from_v(y_t, t, model_output)
        else:
            # Model predicts noise ε, compute x₀_hat from noise
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

    def predict_start_from_v(self, y_t, t, v):
        """
        从v参数化预测初始数据
        """
        sqrt_alpha_t = extract(torch.sqrt(self.gammas), t, y_t.shape)
        sqrt_one_minus_alpha_t = extract(torch.sqrt(1. - self.gammas), t, y_t.shape)
        return sqrt_alpha_t * y_t - sqrt_one_minus_alpha_t * v

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, ddim_sample_steps=50, roofline_image=None):
        """
        使用 DDIM 采样进行图像恢复。
        参数：
            y_cond: 已知条件部分
            y_t: 初始噪声（如果没有提供，则随机初始化）
            y_0: 原始图像（用于补全任务）
            mask: 已知区域的掩码（用于补全任务）
            roofline_image: roofline图像
        """
        # 检查预训练模型是否已加载
        if self.Roofdenoise_fn is None or self.RoofLinedenoise_fn is None:
            raise RuntimeError("Pretrained models (roof and roofline) must be loaded before inference. "
                             "Call load_pretrained_models() first.")
        
        device = self.accelerator.device if self.accelerator is not None else y_cond.device
        b, *_ = y_cond.shape
        
        # 构建 DDIM 采样的时间步序列
        ddim_steps = ddim_sample_steps
        timesteps = torch.linspace(0, self.num_timesteps - 1, steps=ddim_steps, dtype=torch.long, device=y_cond.device)
        timesteps = timesteps.flip(0)  # 逆序采样，从高到低
        
        # timesteps_prev
        timesteps_prev = torch.cat([timesteps, torch.tensor([-1], device=device, dtype=torch.long)])
        timesteps_prev = timesteps_prev[1:]  # 去掉第一个
        
        # 初始化噪声
        y_t_roof = default(y_t, lambda: torch.randn_like(y_cond))
        y_t_roof = y_t_roof*mask+(1.-mask)*y_0
        y_t_roofline = default(y_t, lambda: torch.randn_like(roofline_image))
        y_0_hat_roofline = torch.zeros_like(roofline_image)  # 初始化y0_hat_roofline为0
        
        # 保存采样的图像
        ret_arr_roof = [y_t_roof]
        ret_arr_roofline = [y_t_roofline]
        ret_arr_roofline_y0 = [y_0_hat_roofline]
        
        for t_val, t_prev in zip(timesteps, timesteps_prev):
            t_tensor = torch.full((b,), t_val.item(), device=device, dtype=torch.long)
            t_tensor_prev = torch.full((b,), t_prev.item(), device=device, dtype=torch.long)
            
            # 准备MOM输入：上一步的y_t_roof和y_0_hat_roofline（归一化到[0,1]）
            y_0_hat_roofline_norm = (y_0_hat_roofline + 1.) / 2.
            
            # MOM 预测
            mom_roof, mom_roofline = self.MOM(
                torch.cat([y_cond, y_t_roof, y_0_hat_roofline_norm*mask], dim=1), 
                extract(self.gammas, t_tensor, x_shape=(1, 1)).to(device)
            )
            
            # Roofline 采样
            y_t_roofline, y_0_hat_roofline = self.p_sample_ddim_rl(
                y_t_roofline, t_tensor, t_tensor_prev, 
                y_cond=y_cond, eta=0.0, control_image=y_t_roof,self_cond=y_0_hat_roofline_norm
            )
            
            # Roof 采样
            y_t_roof, _ = self.p_sample_ddim_r(
                y_t_roof, t_tensor, t_tensor_prev, 
                y_cond=y_cond, control_image=y_0_hat_roofline
            )
            
            # 如果是补全任务，保留已知区域
            if mask is not None and y_0 is not None:
                y_t_roof = y_0 * (1. - mask) + mask * y_t_roof
                y_t_roofline = roofline_image * (1. - mask) + mask * y_t_roofline
            
            # 保存采样的图像
            ret_arr_roof.append(y_t_roof)
            ret_arr_roofline.append(y_t_roofline)
            ret_arr_roofline_y0.append(y_0_hat_roofline)
        
        # 拼接所有采样的中间结果
        ret_arr_roof = torch.cat(ret_arr_roof, dim=0)
        ret_arr_roofline = torch.cat(ret_arr_roofline, dim=0)
        ret_arr_roofline_y0 = torch.cat(ret_arr_roofline_y0, dim=0)
        
        return y_t_roof, ret_arr_roof, ret_arr_roofline, ret_arr_roofline_y0, y_0_hat_roofline



    def forward(self, y_0, y_cond=None, mask=None, noise=None,roofline_image=None): #训练过程
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

        noise_roof = default(noise, lambda: torch.randn_like(y_0)) #heightmap的噪声
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_roof) #加噪后的图像
        
        noise_roofline = default(noise, lambda: torch.randn_like(roofline_image)) #roofline的噪声
        y_roofline_noisy = self.q_sample(
            y_0=roofline_image, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_roofline) #加噪后的roofline图像

        if mask is not None:
            #noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,control_image)
            with torch.no_grad(): # This no_grad() is for the first roofline prediction which is not dependent on MOM
                roofline_output1 = self.RoofLinedenoise_fn(torch.cat([y_cond, y_roofline_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,y_noisy,self_cond=torch.zeros_like(y_noisy)) #从y_cond和y_roofline_noisy中去噪声，通过unet
                if self.parameterization == "v":
                    y_roofline_0_hat_selfcond = self.predict_start_from_v(y_roofline_noisy, t, roofline_output1)
                else:
                    # 如果不是v参数化，使用默认的噪声预测方法
                    y_roofline_0_hat_selfcond = self.predict_start_from_noise_with_gamma(y_roofline_noisy, sample_gammas, roofline_output1)
            

            y_roofline_0_hat_selfcond_norm = (y_roofline_0_hat_selfcond + 1.) / 2.
            
            mom_roof,mom_roofline = self.MOM(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0,y_roofline_0_hat_selfcond_norm*mask+(1.-mask)], dim=1), sample_gammas,) #从y_cond和y_noisy中去噪声，通过unet

            # Removed torch.no_grad() here
            roofline_output2 = self.RoofLinedenoise_fn(torch.cat([y_cond, y_roofline_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas, mom_roof,self_cond=roofline_output1) #从y_cond和y_roofline_noisy中去噪声，通过unet
            roof_output = self.Roofdenoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas,mom_roofline) #从y_cond和y_noisy中去噪声，通过unet
            
            # The prediction of y_roofline_y0_hat should also not be in no_grad() if roofline_output2 requires grad
            if self.parameterization == "v":
                y_roofline_y0_hat = self.predict_start_from_v(y_roofline_noisy, t, roofline_output2)
            else:
                y_roofline_y0_hat = self.predict_start_from_noise_with_gamma(y_roofline_noisy, sample_gammas, roofline_output2)
            

            if hasattr(self.loss_fn, '__name__') and self.loss_fn.__name__ in ['masked_mse_loss', 'masked_l1_loss', 'combined_loss_with_masked_ssim']:
                #loss = self.loss_fn(mask*noise, mask*noise_hat, mask) # might not be necessary 
                loss_roof_noise = self.loss_fn(mask*noise_roof, mask*roof_output, mask) # might not be necessary
                loss_roofline_noise = self.loss_fn(mask*noise_roofline, mask*roofline_output2, mask) # might not be necessary
                loss_roofline_recon = self.loss_fn(mask*roofline_image, mask* y_roofline_y0_hat, mask) # might not be necessary
                loss = loss_roof_noise + loss_roofline_noise + loss_roofline_recon
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas) #从y_cond和y_noisy中去噪声，通过unet
            #mask不存在时，直接去噪声，不需要保留mask部分。也就是说此时的y_noisy在mask外的部分也有噪声
            loss = self.loss_fn(noise, noise_hat)
            # 当mask为None时，设置默认值
            loss_roof_noise = loss
            loss_roofline_noise = loss
            loss_roofline_recon = loss
            y_roofline_y0_hat = None
        return loss,{"loss_roof_noise":loss_roof_noise,"loss_roofline_noise":loss_roofline_noise,"loss_roofline_recon":loss_roofline_recon} ,y_roofline_y0_hat#返回损失和其他信息，方便后续的评估和保存模型


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


