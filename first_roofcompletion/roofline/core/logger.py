import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import cv2

import core.util as Util

class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])  #phase is train or test，to retrieve the logger
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):   #当访问的属性不存在时，会调用该方法
        if self.rank != 0: # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns: #如果name在infologger_ftns中，返回logger的相应函数或属性self.loger.name(info, *args, **kwargs)
            print_info = getattr(self.logger, name, None) # get the function or attribute of logger, if not exist, return None
            def wrapper(info, *args, **kwargs): #包装函数
                print_info(info, *args, **kwargs) #绑定到print_info函数，传入info参数
            return wrapper
            #作用是向log中追加一条记录

    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime) s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')  # append to the file一个句柄
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler() #发送到控制台
            sh.setFormatter(formatter)
            l.addHandler(sh)

class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        self.use_color_map = opt['use_color_map']
        self.out_type = np.dtype(opt['out_type'])
        self.scale_factor = opt['scale_factor']

        if enabled and self.rank==0: #在tensorboard中只有rank=0的进程才会启用记录
            log_dir = str(log_dir) # 把log_dir转换为字符串

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]: #尝试两个tensorboard模块
                try:  #将self.writer设置为tensorboard模块的SummaryWriter对象。支持add_scalar、add_scalars、add_image、add_images等函数
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir) #导入tensorboard模块并创建SummaryWriter对象
                    succeeded = True
                    break
                except ImportError: #如果导入失败，输出警告
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results):
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        ''' get names and corresponding images from results[OrderedDict] '''
        try:
            names = results['name']
            outputs = Util.postprocess(results['result'], out_type=self.out_type, scale_factor=self.scale_factor)
            for i in range(len(names)):
                output_path = os.path.join(result_path, names[i])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if self.use_color_map and len(outputs[i].shape) == 2:
                    colorized_img = cv2.applyColorMap(outputs[i], cv2.COLORMAP_JET)
                    cv2.imwrite(output_path, colorized_img)
                else:                   
                    Image.fromarray(outputs[i]).save(output_path)
        except:
            raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')

    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns: #如果name是支持的tensorboard函数
            add_data = getattr(self.writer, name, None)  # add_data赋值为tensorboard的相应函数，这里是引用传递
            def wrapper(tag, data, *args, **kwargs): #包装函数
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions: #如果name不在tag_mode_exceptions中
                        tag = '{}/{}'.format(self.phase, tag) #tag格式为phase/tag
                    add_data(tag, data, self.iter, *args, **kwargs) #add_scalar(tag, scalar_value, global_step=None, walltime=None)
            return wrapper #返回包装函数
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class LogTracker:
    """
    record training numerical indicators.
    """
    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average']) #len(keys)行 ，3列
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
        # train/loss: 0.1 