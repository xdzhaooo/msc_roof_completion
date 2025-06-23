import os
from matplotlib import pyplot as plt
import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast, GradScaler
gradscaler = GradScaler()

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight) #引用传递，所以ma_model的参数也会被更新
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(BaseModel):
    def __init__(
        self, 
        networks, 
        autoencoder=None,
        losses=None, 
        sample_num=8, 
        task='inpainting', 
        optimizers=None,
        lr_schedulers=None,
        ema_scheduler=None,
        cond_on_mask=False,
        **kwargs
    ):
        ''' must to init BaseModel with kwargs '''
        # 从kwargs中提取BaseModel需要的参数
        super(Trainer, self).__init__(
            kwargs.get('opt'),
            kwargs.get('phase_loader'),
            kwargs.get('val_loader'),
            kwargs.get('metrics'),
            kwargs.get('logger'),
            kwargs.get('writer')
        )
        
        # 手动设置accelerator属性
        self.accelerator = kwargs.get('accelerator')

        # 参数检查和默认值设置
        if losses is None:
            losses = []
        if optimizers is None:
            optimizers = [{'lr': 2e-4, 'weight_decay': 0.0}]

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay']) #初始化EMA类
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        if self.accelerator:
            print("Using accelerator")
            self.netG = self.accelerator.prepare(self.netG) #将网络模型转移到GPU上，如果是多GPU训练，还需要将模型转移到多个GPU上
            if self.ema_scheduler is not None:
                self.netG_EMA = self.accelerator.prepare(self.netG_EMA)
        else:
            self.netG = self.set_device(self.netG, distributed=self.opt['distributed']) #将网络模型转移到GPU上，如果是多GPU训练，还需要将模型转移到多个GPU上
            if self.ema_scheduler is not None:
                self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed']) #将EMA模型转移到GPU上，如果是多GPU训练，还需要将模型转移到多个GPU上
        self.load_networks() #如果有预训练模型，加载预训练模型，否则从头开始训练，不做任何操作

        optimizer_params = optimizers[0].copy()  # 复制字典
        optimizer_params['lr'] = float(optimizer_params['lr'])  # 转换为 float
        optimizer_params['weight_decay'] = float(optimizer_params['weight_decay'])  # 转换为 float

        #self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optG = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizer_params) #创建adam优化器，只优化需要梯度的参数。filter函数返回一个迭代器，只有当p.requires_grad为True时，才会返回p，list函数将迭代器转换为列表。**optimizers[0]是将优化器的参数以字典的形式传入，如lr=0.001，betas=(0.9, 0.999)等
        #创建adam优化器，只优化需要梯度的参数。filter函数返回一个迭代器，只有当p.requires_grad为True时，才会返回p，list函数将迭代器转换为列表。**optimizers[0]是将优化器的参数以字典的形式传入，如lr=0.001，betas=(0.9, 0.999)等
        self.optimizers.append(self.optG) #长度是一致的，并且相同位置上的调度器（scheduler）对应于同一个优化器（optimizer
        
        if self.accelerator:
            self.optG = self.accelerator.prepare(self.optG)
        
        if lr_schedulers is not None:
            self.schedulers.append(LinearLR(self.optG, **lr_schedulers[0]))
            #如果有学习率调整器，为新建的优化器创建学习率调整器，LinearLR是一个线性调整器，**lr_schedulers[0]是将调整器的参数以字典的形式传入，如end_lr=0.0等

        self.resume_training() #测试状态或重启训练，如果有预训练的参数，比如模型、优化器、调度器等；否则从头开始训练，不做任何操作

        if self.accelerator:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(device=self.accelerator.device,phase=self.phase,accelerator=self.accelerator) #设置噪声调度器，传入当前设备和当前阶段
        else:
            if self.opt['distributed']:
                self.netG.module.set_loss(self.loss_fn)
                self.netG.module.set_new_noise_schedule(phase=self.phase)
            else:
                self.netG.set_loss(self.loss_fn) #传入的是一个损失函数对象 set_loss 来自于network.py
                self.netG.set_new_noise_schedule(phase=self.phase)#调用network.py中的set_new_noise_schedule函数，设置gammas，betas等参数

        ''' can rewrite in inherited class for more informations logging '''
        #创建记录器，记录训练过程中的损失值，如
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train') #将list解包为参数列表，也就是一个个参数
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

        self.cond_on_mask = cond_on_mask
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))#使用get方法获取字典中的值，如果没有则返回None
        self.gt_image = self.set_device(data.get('gt_image'))
        self.gt_normal_image = self.set_device(data.get('gt_normal_image'))
        self.gt_height_image = self.set_device(data.get('gt_height_image'))
        self.footprint = self.set_device(data.get('footprint'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
        self.height_range = data.get('height_range')
        self.mid_height = data.get('mid_height')
        self.roofline_img = self.set_device(data.get('roofline_img'))
        
    def get_current_visuals(self, phase='train'):
        dict = {}
        if self.task in ['inpainting','uncropping'] and self.mask is not None and self.mask_image is not None:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(), #detach()复制一个新的tensor，且从当前计算图中分离出来
                'mask_image': (self.mask_image+1)/2, #从[-1,1]转换到[0,1]
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2,
                'roofline_output': (self.roofline_output.detach()[:].float().cpu())
            })
        if phase == 'train':
            dict.update({
                'roofline_y0_hat': (self.roofline_y0_hat.detach()[:].float().cpu())
            })
        dict.update({
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
            'roofline_img': (self.roofline_img.detach()[:].float().cpu()),
        })
        return dict

    def convert_to_colormap(self, image, cmap):
        image = image.squeeze(0).numpy()
        cmapped = plt.get_cmap(cmap)(image)

        # The returned array from the colormap has shape (H, W, 4) (RGBA image).
        # We convert it back to PyTorch tensor and get rid of the alpha channel 
        # assuming you want a (3, H, W) tensor.
        return torch.from_numpy(cmapped[:, :, :3]).permute(2, 0, 1)

    def save_current_results(self):
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return {}
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            mid_height, height_range = self.mid_height[idx], self.height_range[idx]
            mid_height = mid_height.detach().float().cpu()
            height_range = height_range.detach().float().cpu()

            ret_path.append('GT_{}'.format(self.path[idx]))
            gt_image = torch.clamp(self.gt_image[idx], -1, 1)
            gt_img = gt_image.detach().float().cpu()
            gt_img = torch.clamp(gt_img, -1, 1)
            gt_mask = gt_img > -1
            gt_img[gt_mask] = gt_img[gt_mask] * 0.5 * height_range + mid_height
            gt_img[~gt_mask] = 0
            ret_result.append(gt_img)

            ret_path.append('Cond_{}'.format(self.path[idx]))
            cond_img = torch.clamp(self.cond_image[idx], -1, 1)
            cond_img = cond_img.detach().float().cpu()
            cond_mask = cond_img > -1
            cond_img[cond_mask] = cond_img[cond_mask] * 0.5 * height_range + mid_height
            cond_img[~cond_mask] = 0
            ret_result.append(cond_img)

            ret_path.append('roofline_{}'.format(self.path[idx]))
            roofline_img = torch.clamp(self.roofline_img[idx], 0, 1)
            roofline_img = roofline_img.detach().float().cpu()
            # roofline_mask = roofline_img > 0
            # roofline_img[roofline_mask] = 1
            # roofline_img[~roofline_mask] = 0
            ret_result.append(roofline_img)

            ret_path.append(self.path[idx])
            output = torch.clamp(self.output[idx], -1, 1)
            output = output.detach().float().cpu()
            out_mask = output > -0.95
            output[out_mask] = output[out_mask] * 0.5 * height_range + mid_height
            output[~out_mask] = 0
            output[output < 0] = 0
            ret_result.append(output)

            ret_path.append('roofline_output_{}'.format(self.path[idx]))
            roofline_output = torch.clamp(self.roofline_output[idx], 0, 1)
            roofline_output = roofline_output.detach().float().cpu()
            # roofline_output_mask = roofline_output > 0
            # roofline_output[roofline_output_mask] = 1
            # roofline_output[~roofline_output_mask] = 0
            ret_result.append(roofline_output)

            # if self.sample_num > 0:
            # for k in range(self.sample_num + 1):
            #     print('Inter_{}_{}'.format(k, self.path[idx]))
            #     ret_path.append('Inter_{}_{}'.format(k, self.path[idx]))
            #     ret_result.append(self.visuals[k * self.batch_size + idx].detach().float().cpu())
            
            #ddim 50 step, save 50 images
            #save to progress folder
            from torchvision.utils import save_image
            #
            progress_path = os.path.join(self.opt['path']['results'],"roof_denoising_progress", str(self.epoch))
            os.makedirs(progress_path, exist_ok=True)
            save_image((self.visuals[idx:-1:16]+1)/2, os.path.join(progress_path, self.path[idx]))
            rl_progress_path = os.path.join(self.opt['path']['results'],"roofline_progress", str(self.epoch))
            os.makedirs(rl_progress_path, exist_ok=True)
            save_image((self.roofline_arr[idx:-1:16]), os.path.join(rl_progress_path, self.path[idx]))
            rl_y0_progress_path = os.path.join(self.opt['path']['results'],"roofline_y0_progress", str(self.epoch))
            os.makedirs(rl_y0_progress_path, exist_ok=True)
            save_image((self.roofline_y0_arr[idx:-1:16]), os.path.join(rl_y0_progress_path, self.path[idx]))

        
        if self.task in ['inpainting','uncropping'] and self.mask is not None:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask.detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        self.optG.zero_grad()
        accumulation_steps = 1
        i = 0

        from torch.utils.data.distributed import DistributedSampler

        if isinstance(self.phase_loader.sampler, DistributedSampler):
            self.phase_loader.sampler.set_epoch(self.epoch)

        for train_data in tqdm.tqdm(self.phase_loader):
            i += 1
            self.set_input(train_data)
            if self.cond_on_mask:
                mask_channel = self.mask.clone()
                mask_channel[mask_channel == 0] = -1
                cond_image = torch.cat((self.cond_image, mask_channel), dim=1)
            else:
                cond_image = self.cond_image

            roofline_img = self.roofline_img

            with autocast(enabled=self.accelerator.mixed_precision != 'no' if self.accelerator else False):
                # self.gt_image = self.gt_image.to(dtype=torch.float16)
                # cond_image = cond_image.to(dtype=torch.float16)
                # self.mask = self.mask.to(dtype=torch.float16)
                # roofline_img = roofline_img.to(dtype=torch.float16)
                loss, loss_discretize, roofline_y0_hat = self.netG(self.gt_image, cond_image, mask=self.mask, roofline_image=roofline_img)
                raw_loss = loss.clone()
                loss = loss / accumulation_steps
                loss_discretize = {k: v / accumulation_steps for k, v in loss_discretize.items()}

            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            self.roofline_y0_hat = roofline_y0_hat

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.phase_loader):
                if self.accelerator is not None and self.accelerator.mixed_precision != 'no':
                    self.accelerator.scaler.step(self.optG)
                    self.accelerator.scaler.update()
                else:
                    self.optG.step()
                self.optG.zero_grad()
                for scheduler in self.schedulers:
                    scheduler.step()

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, raw_loss.item())

            # 在日志和可视化之前同步所有进程
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            if self.accelerator is None or self.accelerator.is_main_process:
                if self.iter % self.opt['train']['log_iter'] == 0:
                    for key, value in self.train_metrics.result().items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals().items():
                        self.writer.add_images(key, value)
                    self.writer.add_scalar('Learning Rate', self.optG.param_groups[0]['lr'])
                    for key, value in loss_discretize.items():
                        self.writer.add_scalar(key, value)
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))

            # 在 EMA 更新之前同步所有进程
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        return self.train_metrics.result()
    
    @torch.no_grad()
    def val_step(self):
        if self.accelerator is not None:
            print(f"Rank {self.accelerator.process_index}: Num processes {self.accelerator.num_processes}")
            rank = self.accelerator.process_index
        else:
            print("Running without accelerator")
            rank = 0
            
        import time
        import traceback
        self.netG.eval()
        self.val_metrics.reset()
        
        print(f"[Rank {rank}] Starting validation step at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 检查 val_loader 初始化
            print(f"[Rank {rank}] Length of val_loader: {len(self.val_loader)}")
            
            for i, val_data in enumerate(tqdm.tqdm(self.val_loader, desc=f"Validation [Rank {rank}]")):
                print(f"[Rank {rank}] Processing batch {i}, data keys: {list(val_data.keys())}")
                
                # 检查 set_input 执行
                try:
                    start_time = time.time()
                    self.set_input(val_data)
                    print(f"[Rank {rank}] set_input for batch {i} completed in {time.time() - start_time:.2f}s")
                    
                    # 检查 height_range
                    if hasattr(self, 'height_range'):
                        print(f"[Rank {rank}] height_range: {self.height_range}, device: {self.height_range.device}")
                    
                except Exception as e:
                    print(f"[Rank {rank}] Error in set_input for batch {i}: {str(e)}")
                    print(f"[Rank {rank}] Traceback: {traceback.format_exc()}")
                    raise
                
                if self.cond_on_mask:
                    mask_channel = self.mask.clone()
                    mask_channel[mask_channel == 0] = -1
                    cond_image = torch.cat((self.cond_image, mask_channel), dim=1)
                else:
                    cond_image = self.cond_image

                # 记录推理开始时间
                start_time = time.time()
                self.output, self.visuals, self.roofline_arr, self.roofline_y0_arr, self.roofline_output = self.netG.restoration(
                    cond_image, y_t=None, y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, roofline_image=self.roofline_img
                )
                print(f"[Rank {rank}] Inference for batch {i} completed in {time.time() - start_time:.2f}s")

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                print(f"[Rank {rank}] Reaching sync point for batch {i}")
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                print(f"[Rank {rank}] Passed sync point for batch {i}")

                # 只在主进程或无accelerator时执行
                if self.accelerator is None or self.accelerator.is_main_process:
                    for met in self.metrics:
                        key = met.__name__
                        value = met(self.gt_image, self.output)
                        self.val_metrics.update(key, value)
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals(phase='val').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results())

        except Exception as e:
            print(f"[Rank {rank}] Exception occurred: {str(e)}")
            print(f"[Rank {rank}] Traceback: {traceback.format_exc()}")
            raise

        print(f"[Rank {rank}] Validation step completed")
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        return self.val_metrics.result()
    
    def test(self):
        self.netG.eval()
        self.test_metrics.reset()

        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)

                if self.cond_on_mask:
                    mask_channel = self.mask.clone()
                    mask_channel[mask_channel == 0] = -1
                    cond_image = torch.cat((self.cond_image, mask_channel), dim=1)
                else:
                    cond_image = self.cond_image

                self.output, self.visuals = self.netG.restoration(
                    cond_image, y_t=None, y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num
                )

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')

                if self.accelerator is None or self.accelerator.is_main_process:
                    for met in self.metrics:
                        key = met.__name__
                        value = met(self.gt_image, self.output)
                        self.test_metrics.update(key, value)
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results())

            test_log = self.test_metrics.result()
            test_log.update({'epoch': self.epoch, 'iters': self.iter})

            if self.accelerator is None or self.accelerator.is_main_process:
                for key, value in test_log.items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        #在config.json如果设置了resume_state参数，就会加载预训练模型的权重，否则会从头开始训练，不输入resume_state权重
        # if self.opt['distributed']:
        #     netG_label = self.netG.module.__class__.__name__ 
        #     #如果采用分布式训练（self.opt['distributed'] 为 True），生成器网络 netG 通常被包装在一个 DistributedDataParallel 对象中，此时实际的模型在 self.netG.module 里，所以用 self.netG.module.__class__.__name__ 来获得模型的类名作为标签。
        # else:
        #     netG_label = self.netG.__class__.__name__
        #     #如果不采用分布式训练，直接用 self.netG.__class__.__name__ 来获得模型的类名作为标签。
        # self.load_network(network=self.netG, network_label=netG_label, strict=False)
        # if self.ema_scheduler is not None:
        #     self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)
        if self.accelerator:
            netG_label = self.accelerator.unwrap_model(self.netG).__class__.__name__
            self.load_network(network=self.netG, network_label=netG_label, strict=False)
            if self.ema_scheduler is not None:
                self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)
        else:
            if self.opt['distributed']:
                netG_label = self.netG.module.__class__.__name__
                self.load_network(network=self.netG, network_label=netG_label, strict=False)
            else:
                netG_label = self.netG.__class__.__name__
                self.load_network(network=self.netG, network_label=netG_label, strict=False)
            if self.ema_scheduler is not None:
                self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        # """ load pretrained model and training state. """
        # if self.opt['distributed']:
        #     netG_label = self.netG.module.__class__.__name__
        # else:
        #     netG_label = self.netG.__class__.__name__
        # self.save_network(network=self.netG, network_label=netG_label)
        # if self.ema_scheduler is not None:
        #     self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        # self.save_training_state()
        if self.accelerator is None or self.accelerator.is_main_process:
            if self.accelerator is not None:
                netG_label = self.accelerator.unwrap_model(self.netG).__class__.__name__
                self.save_network(network=self.netG, network_label=netG_label)
                if self.ema_scheduler is not None:
                    self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
            else:
                if self.opt['distributed']:
                    netG_label = self.netG.module.__class__.__name__
                    self.save_network(network=self.netG, network_label=netG_label)
                else:
                    netG_label = self.netG.__class__.__name__
                    self.save_network(network=self.netG, network_label=netG_label)
                if self.ema_scheduler is not None:
                    self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
            self.save_training_state()
        
