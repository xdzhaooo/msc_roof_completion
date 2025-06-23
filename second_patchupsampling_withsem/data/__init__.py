from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.praser import init_obj


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])#为util.set_seed函数设置默认参数，目的是为了在多线程中使用，设置不同的seed，可以保证每个线程的随机数不同，并且保证每次运行结果一致

    phase_dataset, val_dataset = define_dataset(logger, opt) #返回数据集和验证集的对象，但是并没有加载数据，加载数据是在DataLoader中进行的

    '''create datasampler'''
    data_sampler = None
                        #在这段代码中，global_rank 和 world_size 用于设置分布式数据采样器 DistributedSampler，确保数据在多进程训练中均匀分布并避免重复。
    if opt['distributed']: 
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
        #数据分发由sampler控制，不再由dataLoader控制，所以shuffle=False
    
    ''' create dataloader and validation dataloader '''
    #dataloader会自动调用dataset的__getitem__方法(返回一个）和__len__方法，返回一个batch的数据和长度
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) 
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset') #返回一个读取dataset的类的实例或函数的实例
    val_opt = opt['datasets'][opt['phase']]['which_dataset']
    val_opt["name"]=["data.datasetWholePatch", "RoofPatchDataset"]
    val_dataset = init_obj(val_opt, logger, default_file_name='data.dataset', init_type='Dataset') if opt['phase'] == 'train' else None

    valid_len = 0
    patchdata_len = len(phase_dataset)
    data_len = len(val_dataset)
    if 'debug' in opt['name']:  # debug mode，需要在config.json中设置debug_split参数，并在config.json中的name中加入debug
        debug_split = opt['debug'].get('debug_split', 1.0) # debug_split is 1.0 when debug_split is not set.
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
    valid_split = dataloder_opt.get('validation_split', 0)    
    
    ''' divide validation dataset, valid_split==0 when phase is test or validation_split is 0. '''
    if valid_split > 0.0 or 'debug' in opt['name']: 
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        data_len -= valid_len
        useless_dataset, val_dataset = subset_split(dataset=val_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt['seed']))
    
    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], patchdata_len))
    if opt['phase'] == 'train':
        logger.info('Dataset for {} have {} samples.'.format('val', valid_len))   
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist() #返回一个长度为sum(lengths)的随机排列的tensor，然后转换为list
    # indices = list(range(sum(lengths)))
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths): #累积求和[100,120],lengths[100,20]，zip作用是将两个列表中的元素一一对应
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))#0-100,100-120
    return Subsets


# def define_dataset(logger, opt):
#     ''' loading Dataset() class from given file's name '''
#     dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
#     phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset') #返回一个读取dataset的类的实例或函数的实例
#     val_dataset = None

#     valid_len = 0
#     data_len = len(phase_dataset)
#     if 'debug' in opt['name']:  # debug mode，需要在config.json中设置debug_split参数，并在config.json中的name中加入debug
#         debug_split = opt['debug'].get('debug_split', 1.0) # debug_split is 1.0 when debug_split is not set.
#         if isinstance(debug_split, int):
#             data_len = debug_split
#         else:
#             data_len *= debug_split

#     dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
#     valid_split = dataloder_opt.get('validation_split', 0)    
    
#     ''' divide validation dataset, valid_split==0 when phase is test or validation_split is 0. '''
#     if valid_split > 0.0 or 'debug' in opt['name']: 
#         if isinstance(valid_split, int):
#             assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
#             valid_len = valid_split
#         else:
#             valid_len = int(data_len * valid_split)
#         data_len -= valid_len
#         phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt['seed']))
    
#     logger.info('Dataset for {} have {} samples.'.format(opt['phase'], data_len))
#     if opt['phase'] == 'train':
#         logger.info('Dataset for {} have {} samples.'.format('val', valid_len))   
#     return phase_dataset, val_dataset

# def subset_split(dataset, lengths, generator):
#     """
#     split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
#     """
#     indices = randperm(sum(lengths), generator=generator).tolist() #返回一个长度为sum(lengths)的随机排列的tensor，然后转换为list
#     # indices = list(range(sum(lengths)))
#     Subsets = []
#     for offset, length in zip(np.add.accumulate(lengths), lengths): #累积求和[100,120],lengths[100,20]，zip作用是将两个列表中的元素一一对应
#         if length == 0:
#             Subsets.append(None)
#         else:
#             Subsets.append(Subset(dataset, indices[offset - length : offset]))#0-100,100-120
#     return Subsets
