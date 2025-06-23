import os
from abc import abstractmethod
from functools import partial
import collections

import torch
import torch.nn as nn


import core.util as Util
CustomResult = collections.namedtuple('CustomResult', 'name result') #创建一个元组类型CustomResult，包含两个属性，name和result

class BaseModel():
    def __init__(self, opt, phase_loader, val_loader, metrics, logger, writer, accelerator=None):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.opt = opt #来自which_model
        self.phase = opt['phase']
        # 设置set_device函数，兼容无accelerator的情况
        if 'accelerator' in opt and opt['accelerator'] is not None and hasattr(opt['accelerator'], 'local_process_index'):
            self.set_device = partial(Util.set_device, rank=opt['accelerator'].local_process_index)
        else:
            self.set_device = partial(Util.set_device, rank=0)
        self.accelerator = accelerator if accelerator is not None else None #如果没有加速器，则设置为None
        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []

        ''' process record '''
        self.batch_size = self.opt['datasets'][self.phase]['dataloader']['args']['batch_size']
        self.epoch = 0
        self.iter = 0 

        if self.accelerator is not None:
            self.phase_loader, self.val_loader = self.accelerator.prepare(phase_loader, val_loader) #如果有加速器，则将数据加载器传入加速器中
        else:
            self.phase_loader = phase_loader
            self.val_loader = val_loader
        self.metrics = metrics

        ''' logger to log file, which only work on GPU 0. writer to tensorboard and result file '''
        self.logger = logger
        self.writer = writer
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}

    def train(self):
        print(self.accelerator.is_main_process if self.accelerator else True, "accelerator.is_main_process")
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
        if self.val_loader is None:
            self.logger.warning('Validation stop where dataloader is None, Skip it.')
            if self.accelerator is None or self.accelerator.is_main_process:
                self.logger.warning('Validation stop where dataloader is None, Skip it.')
        else:
            print('start val_step')
            val_log = self.val_step()
            if self.accelerator is None or self.accelerator.is_main_process:
                for key, value in val_log.items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
        if self.accelerator is None or self.accelerator.is_main_process:
            self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        
        n_epoch = int(self.opt['train']['n_epoch'])
        n_iter = int(self.opt['train']['n_iter'])
        #while self.epoch <= self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
        while self.epoch < n_epoch and self.iter < n_iter:
            self.epoch += 1
            if self.opt['distributed'] and self.accelerator is not None:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) #确保每个epoch下，不同的进程使用不同的随机顺序不重复

            train_log = self.train_step() #实现在子类中，这是训练的核心步骤

            ''' save logged informations into log dict ''' 
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard ''' 
            if self.accelerator is None or self.accelerator.is_main_process:
                for key, value in train_log.items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()

            if self.epoch % self.opt['train']['val_epoch'] == 0:
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')
                else:
                    val_log = self.val_step()
                    if self.accelerator is None or self.accelerator.is_main_process:
                        for key, value in val_log.items():
                            self.logger.info('{:5s}: {}\t'.format(str(key), value))
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.')

    def test(self):
        pass

    @abstractmethod #一个抽象方法，必须在子类中实现
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    def test_step(self):
        pass
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        # if self.opt['global_rank'] !=0:
        #     return
        # if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        #     network = network.module
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        # if self.opt['global_rank'] !=0:
        #     return
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        save_filename = '{}_{}.pth'.format(self.epoch, network_label)
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        if self.accelerator:
            network = self.accelerator.unwrap_model(network) #如果使用了加速器，则将模型解包,否则直接使用模型
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        """Load pretrained model weights.
        
        Args:
            network: The network to load weights into
            network_label: Label for the network (used in logging)
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        if self.opt['path']['resume_path'] is None:
            return
            
        model_path = self.opt['path']['resume_path']
        if not os.path.exists(model_path):
            if self.accelerator is not None and self.accelerator.is_main_process:
                self.logger.warning(f'Pretrained model not found: {model_path}')
            return
            
        if self.accelerator is not None and self.accelerator.is_main_process:
            self.logger.info(f'Loading pretrained model from {model_path}')
        
        # Handle DataParallel and DistributedDataParallel
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
            
        # Unwrap model if using accelerator
        if self.accelerator is not None:
            network = self.accelerator.unwrap_model(network)
            
        # Load state dict
        state_dict = torch.load(model_path, map_location=lambda storage, loc: Util.set_device(storage))
        
        # Load weights
        network.load_state_dict(state_dict, strict=strict)
        
        if self.accelerator is not None and self.accelerator.is_main_process:
            self.logger.info(f'Successfully loaded pretrained model from {model_path}')

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """
        # if self.opt['global_rank'] !=0:
        #     return
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(self.epoch)
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase!='train' or self. opt['path']['resume_state'] is None:
            return
        self.logger.info('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_path = "{}.state".format(self. opt['path']['resume_state'])
        
        if not os.path.exists(state_path):
            self.logger.warning('Training state in [{:s}] is not existed, Skip it'.format(state_path))
            return

        self.logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: self.set_device(storage))
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers {} != {}'.format(len(resume_schedulers), len(self.schedulers))
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    def load_everything(self, network, network_label, strict=True):
        """Alias for load_network for backward compatibility"""
        return self.load_network(network, network_label, strict)
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')
    
