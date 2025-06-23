import random
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid


def tensor2img(tensor, min_max=(-1, 1), out_type=np.uint8, scale_factor=1):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    # tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    elif out_type == np.uint16:
        # img_np = ((img_np+1) / 2 * scale_factor * 256).round()
        img_np = (img_np * scale_factor * 256).round()

    return img_np.astype(out_type).squeeze()

def postprocess(images, out_type=np.uint8, scale_factor=1):
	return [tensor2img(image, out_type=out_type, scale_factor=scale_factor) for image in images]


def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible    
		torch.backends.cudnn.deterministic = True      # 保证每次结果一样，确定性
		torch.backends.cudnn.benchmark = False        #不启用自动寻找最适合当前配置的高效算法，会减慢速度，但是可以保证结果一样
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False       
		torch.backends.cudnn.benchmark = True

def set_gpu(args, distributed=False, rank=0): #将传入参数设置为gpu或ddp，可以是一个模型、张量或其他需要转移到 GPU 上的对象。
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
#移动到gpu上，device_ids指定gpu，output_device指定输出的gpu，每次进入前向传播前就会进行广播，确保所有设备使用一致的 buffer （参数）数据；自动发现未使用的参数，减少内存占用		
#ddp中，通常会定义好设备，device = torch.device(f"cuda:{local_rank}")
	
		
def set_device(args, distributed=False, rank=0):    #basemodel中传入的rank参数是opt['global_rank']
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)
	return args



