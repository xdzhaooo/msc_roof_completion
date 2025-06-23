import os
from matplotlib import pyplot as plt
import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

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

class Palette(BaseModel):
    def __init__(
        self, 
        networks, 
        losses, 
        sample_num, 
        task, 
        optimizers,
        lr_schedulers=None,
        ema_scheduler=None,
        cond_on_mask=False,
        **kwargs
    ):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)
        #super().__init__(**kwargs)
        #调用父类的__init__函数，传入kwargs参数

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
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed']) #将网络模型转移到GPU上，如果是多GPU训练，还需要将模型转移到多个GPU上
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed']) #将EMA模型转移到GPU上，如果是多GPU训练，还需要将模型转移到多个GPU上
        self.load_networks() #如果有预训练模型，加载预训练模型，否则从头开始训练，不做任何操作

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        #创建adam优化器，只优化需要梯度的参数。filter函数返回一个迭代器，只有当p.requires_grad为True时，才会返回p，list函数将迭代器转换为列表。**optimizers[0]是将优化器的参数以字典的形式传入，如lr=0.001，betas=(0.9, 0.999)等
        self.optimizers.append(self.optG) #长度是一致的，并且相同位置上的调度器（scheduler）对应于同一个优化器（optimizer
        if lr_schedulers is not None:
            self.schedulers.append(LinearLR(self.optG, **lr_schedulers[0]))
            #如果有学习率调整器，为新建的优化器创建学习率调整器，LinearLR是一个线性调整器，**lr_schedulers[0]是将调整器的参数以字典的形式传入，如end_lr=0.0等

        self.resume_training() #测试状态或重启训练，如果有预训练的参数，比如模型、优化器、调度器等；否则从头开始训练，不做任何操作

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


        # conv_out = self.netG.denoise_fn.out[2]  # 或者 model.unet.out[2]，取决于你的结构
        # nn.init.normal_(conv_out.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(conv_out.bias)
        
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
        self.control_image = self.set_device(data.get('control_image'))
        
    def get_current_visuals(self, phase='train'):
        dict = {}
        if self.task in ['inpainting','uncropping'] and self.mask is not None and self.mask_image is not None:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(), #detach()复制一个新的tensor，且从当前计算图中分离出来
                'mask_image': (self.mask_image+1)/2, #从[-1,1]转换到[0,1]
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2, #从[-1,1]转换到[0,1]
            })
        if phase == 'train':
            dict.update({
                'y0_hat': (self.y0_hat.detach()[:].float().cpu()+1)/2,  
            })
        dict.update({
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
            'control_image': (self.control_image.detach()[:].float().cpu()+1)/2,

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
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            mid_height, height_range = self.mid_height[idx], self.height_range[idx]

            ret_path.append('GT_{}'.format(self.path[idx]))
            gt_image = torch.clamp(self.gt_image[idx], -1, 1)
            gt_img = gt_image.detach().float().cpu()
            #gt_img = torch.clamp(gt_img, 0, 1)
            gt_img = (gt_img + 1) / 2
            gt_mask = gt_img >= 0
            #gt_img[gt_mask] = gt_img[gt_mask] * 0.5 * height_range + mid_height
            gt_img[~gt_mask] = 0
            ret_result.append(gt_img)

            ret_path.append('Cond_{}'.format(self.path[idx]))
            cond_img = torch.clamp(self.cond_image[idx], -1, 1)
            cond_img = cond_img.detach().float().cpu()
            cond_mask = cond_img > -1
            cond_img[cond_mask] = cond_img[cond_mask] * 0.5 * height_range + mid_height
            cond_img[~cond_mask] = 0
            ret_result.append(cond_img)
            ret_path.append('Control_{}'.format(self.path[idx]))
            control_img = torch.clamp(self.control_image[idx], -1, 1) #将tensor中的值限制在[-1,1]之间
            control_img = control_img.detach().float().cpu()
            control_mask = control_img > -1
            control_img[control_mask] = 1
            control_img[~control_mask] = 0
            ret_result.append(control_img)

            ret_path.append(self.path[idx])
            output = torch.clamp(self.output[idx], -1, 1)
            output = output.detach().float().cpu()
            output = (output + 1) / 2
            out_mask = output >= 0
            # output[out_mask] = output[out_mask] * 0.5 * height_range + mid_height
            output[~out_mask] = 0
            output[output < 0] = 0
            ret_result.append(output)

            # if self.sample_num > 0:
            # for k in range(self.sample_num + 1):
            #     print('Inter_{}_{}'.format(k, self.path[idx]))
            #     ret_path.append('Inter_{}_{}'.format(k, self.path[idx]))
            #     ret_result.append(self.visuals[k * self.batch_size + idx].detach().float().cpu())
            
            #ddim 50 step, save 50 images
            #save to progress folder
            from torchvision.utils import save_image
            #
            progress_path = os.path.join(self.opt['path']['results'],"denoising_progress", str(self.epoch))
            os.makedirs(progress_path, exist_ok=True)
            save_image((self.visuals[idx:-1:16]+1)/2, os.path.join(progress_path, self.path[idx]))
            y0_path = os.path.join(self.opt['path']['results'],"y0", str(self.epoch))
            os.makedirs(y0_path, exist_ok=True)
            save_image((self.y0_hats[idx:-1:16]+1)/2, os.path.join(y0_path, self.path[idx]))

        
        if self.task in ['inpainting','uncropping'] and self.mask is not None:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path]) #将mask的路径添加到ret_path中
            ret_result.extend(self.mask.detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train() #设置为训练模式
        self.train_metrics.reset() #重置metrics的logger
        #tiduleiji
        self.optG.zero_grad() #梯度清零

        accumulation_steps = 4

        i=0

        for train_data in tqdm.tqdm(self.phase_loader): #train_data是一个字典，包含了训练数据，一个batch，tqdm是一个进度条库
            i+=1
            self.set_input(train_data)
            #self.optG.zero_grad() #梯度清零
            if self.cond_on_mask:   #只有mask内部需要补全的任务才会用到这个参数
                mask_channel = self.mask.clone()
                mask_channel[mask_channel == 0] = -1   #将mask中的0值替换为-1
                cond_image = torch.cat((self.cond_image, mask_channel), dim=1) #将mask_channel和cond_image拼接在一起
            else:
                cond_image = self.cond_image

            control_image = self.control_image
            
            #randomly set the control image and condition image to 0
            p_random = 0.6
            #generate a random number between 0 and and 1 with the same length as the control image and condition image
            rand_control = torch.rand(control_image.shape[0],device=control_image.device)
            rand_cond = torch.rand(cond_image.shape[0],device=cond_image.device)
            #set the control image and condition image to 0 if the random number is less than p_random
            contrl_mask = (rand_control>p_random).float().view(-1,1,1,1)
            cond_mask = (rand_cond>0.1).float().view(-1,1,1,1)
            control_image = control_image * contrl_mask
            cond_image = cond_image * cond_mask
            self.control_image = control_image
            self.cond_image = cond_image



            loss,self.y0_hat = self.netG(self.gt_image, cond_image, mask=self.mask,control_image=self.control_image) #这里的mask就是footprint
            raw_loss = loss.clone()
            loss = loss/accumulation_steps
            #loss.backward()
            gradscaler.scale(loss).backward()
            #self.optG.step()#优化器更新参数
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.phase_loader):  
                # self.optG.step()  # Update parameters
                # self.optG.zero_grad()  # Clear accumulated gradients

                gradscaler.step(self.optG)
                gradscaler.update()
                self.optG.zero_grad()
                for scheduler in self.schedulers:
                    scheduler.step() #更新学习率

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train') #VisualWriter,logger.py中的VisualWriter类，设置当前的epoch和iter，用于tensorboard的记录
            self.train_metrics.update(self.loss_fn.__name__, raw_loss.item()) #loss张量默认有一个item()方法，返回一个Python数值
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items(): #类似phase/metric: value；train/loss: 0.1 
                    self.logger.info('{:5s}: {}\t'.format(str(key), value)) #记录格式化的日志信息：'train/loss: 0.1'
                    self.writer.add_scalar(key, value) #将训练过程中的损失值写入tensorboard
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value) #将训练过程中的图像写入tensorboard
                # Log learning rate
                self.writer.add_scalar('Learning Rate', self.optG.param_groups[0]['lr']) #记录学习率
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0: #迭代次数大于ema_start并且迭代次数是ema_iter的整数倍
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        return self.train_metrics.result()
    
    @torch.no_grad()
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)

                if self.cond_on_mask:
                    mask_channel = self.mask.clone()
                    mask_channel[mask_channel == 0] = -1
                    cond_image = torch.cat((self.cond_image, mask_channel), dim=1)
                else:
                    cond_image = self.cond_image

                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals, self.y0_hats = self.netG.module.restoration(cond_image, y_t=None, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,control_image=self.control_image)
                    else:
                        self.output, self.visuals, self.y0_hats = self.netG.module.restoration(cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals, self.y0_hats = self.netG.restoration(cond_image, y_t=None, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,control_image=self.control_image)
                    else:
                        self.output, self.visuals, self.y0_hats = self.netG.restoration(cond_image, sample_num=self.sample_num)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

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

                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(cond_image, y_t=None,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,control_image=self.control_image)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(cond_image, y_t=None, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,control_image=self.control_image)
                    else:
                        self.output, self.visuals = self.netG.restoration(cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        #在config.json如果设置了resume_state参数，就会加载预训练模型的权重，否则会从头开始训练，不输入resume_state权重
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__ 
            #如果采用分布式训练（self.opt['distributed'] 为 True），生成器网络 netG 通常被包装在一个 DistributedDataParallel 对象中，此时实际的模型在 self.netG.module 里，所以用 self.netG.module.__class__.__name__ 来获得模型的类名作为标签。
        else:
            netG_label = self.netG.__class__.__name__
            #如果不采用分布式训练，直接用 self.netG.__class__.__name__ 来获得模型的类名作为标签。
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

