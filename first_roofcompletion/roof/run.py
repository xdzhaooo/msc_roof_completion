import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')

def main_worker(gpu, ngpus_per_node, opt):
    """Threads running on each GPU"""
    if 'local_rank' not in opt:  # If no local_rank (GPU process index), set local_rank and global_rank to the same value
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))  # Set current GPU, local_rank is both GPU and process index
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl',  # Initialize process group, nccl is the backend protocol for multi-GPU communication
            init_method = opt['init_method'],  # env or tcp mode, currently tcp
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    # Set seed and cuDNN environment
    torch.backends.cudnn.enabled = True  # Enable cuDNN acceleration for CNNs
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    # Set logger
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    # Set networks and dataset
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    # Set metrics, loss, optimizer and schedulers
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/roof_completion.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-o', '--output_dir_name', default=None)
    parser.add_argument('-pt', '--preprocess_type', default=None)
    parser.add_argument('-rs', '--resume_state', default=None)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--footprint_root', default=None)
    parser.add_argument('--mask_root', default=None)
    parser.add_argument('--use_color_map', type=bool, default=False)
    parser.add_argument('--out_type', type=str, default='uint16')
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--n_timestep', type=int, default=None)
    parser.add_argument('--sample_num', type=int, default=0)
    parser.add_argument('--use_footprint', action='store_true', default=None)

    # Parse configs
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    # CUDA devices
    # The GPUs to use. If gpu_ids is None, gpu_str will cause an error, so set gpu_ids in the config file.
    # Set gpu_ids in the config file, 0 for single GPU, 0,1 for two GPUs
    gpu_str = ','.join(str(x) for x in opt['gpu_ids']) 
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str  # GPUs that can be accessed
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    # Use DistributedDataParallel(DDP) and multiprocessing for multi-GPU training
    # [Todo]: multi GPU on multi machine
    # This is for multi-GPU training on a single machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node    # Total number of GPUs, i.e., total number of processes, for single machine multi-GPU
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))   # Start multiple processes, each will call main_worker
        # processing, processing numbers, arguments for each processing, process is automatically numbered as gpu argument for main_worker
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)




#
# Node - Process
# global_rank: 0,1,2,3,4,5,6,7   
# we have 8 GPUs in total, 4 GPUs on each node
# node0-local_rank: 0,1,2,3
# node1-local_rank: 0,1,2,3
#