import os
from collections import OrderedDict
import json
from pathlib import Path
from datetime import datetime
from functools import partial
import importlib
from types  import FunctionType
import shutil
import yaml
def init_obj(opt, logger, *args, default_file_name='default file', given_module=None, init_type='Network', **modify_kwargs):
    """
    finds a function handle with the name given as 'name' in config,
    and returns the instance initialized with corresponding args.
    返回一个读取相应阶段文件的类的实例或函数的实例
    """ 
    if opt is None or len(opt)<1:
        logger.info('Option is None when initialize {}'.format(init_type))
        return None
    
    ''' default format is dict with name key '''
    if isinstance(opt, str):
        opt = {'name': opt}
        logger.warning('Config is a str, converts to a dict {}'.format(opt))

    name = opt['name']
    ''' name can be list, indicates the file and class name of function '''
    if isinstance(name, list):
        file_name, class_name = name[0], name[1]
    else:
        file_name, class_name = default_file_name, name
    try:
        if given_module is not None:
            module = given_module
        else:
            module = importlib.import_module(file_name)  #动态导入模块，从文件名导入模块


        print("模块名：",module,"类名：",class_name) # 打印模块名和类名 
        attr = getattr(module, class_name) # 将模块中的类或函数赋值给attr
        kwargs = opt.get('args', {}) # get the args of opt，如果没有args，则返回空字典
        kwargs.update(modify_kwargs) # update the args with modify_kwargs，modify_kwargs是额外的参数，格式为字典
        ''' import class or function with args '''
        if isinstance(attr, type):  #判断attr是否是一个类
            ret = attr(*args, **kwargs) #实例化类，*args表示任意多个参数，**kwargs表示关键字参数
            ret.__name__  = ret.__class__.__name__ #设置实例的名字
        elif isinstance(attr, FunctionType): 
            ret = partial(attr, *args, **kwargs)
            ret.__name__  = attr.__name__
            # ret = attr
        logger.info('{} [{:s}() form {:s}] is created.'.format(init_type, class_name, file_name))
    except ImportError as e:
        raise NotImplementedError('{} [{:s}() from {:s}] not recognized.'.format(init_type, class_name, file_name))
        print('ImportError:', e)
    return ret


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    """ dict to string for logger """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse(args):
    #json_str = ''
    # with open(args.config, 'r', encoding="utf-8") as f:
    #     for line in f:
    #         line = line.split('//')[0] + '\n'
    #         json_str += line
    # opt = json.loads(json_str, object_pairs_hook=OrderedDict)
            # 加载 YAML 配置文件
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
    # print("Raw config optimizers:", opt['model']['which_model']['args']['optimizers'])
    # print("type of lr:", type(opt['model']['which_model']['args']['optimizers'][0]['lr']))

    ''' replace the config context using args '''
    opt['phase'] = args.phase
    if args.gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in args.gpu_ids.split(',')]
    if args.batch is not None:
        opt['datasets'][opt['phase']]['dataloader']['args']['batch_size'] = args.batch
    if args.resume_state is not None:
        opt["path"]["resume_state"] = args.resume_state
    if args.data_root is not None:
        opt["datasets"][opt['phase']]["which_dataset"]["args"]["data_root"] = args.data_root
    if args.footprint_root is not None:
        opt["datasets"][opt['phase']]["which_dataset"]["args"]["footprint_root"] = args.footprint_root
    if args.mask_root is not None:
        opt["datasets"][opt['phase']]["which_dataset"]["args"]["mask_root"] = args.mask_root
    if args.n_timestep is not None:
        opt["model"]["which_networks"][0]["args"]["beta_schedule"][opt['phase']]["n_timestep"] = args.n_timestep
    if args.sample_num is not None:
        opt["model"]["which_model"]["args"]["sample_num"] = args.sample_num
    if args.use_footprint is not None:
        opt["datasets"][opt['phase']]["which_dataset"]["args"]["use_footprint"] = args.use_footprint

    ''' set cuda environment '''
    if len(opt['gpu_ids']) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    ''' update name '''
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    elif opt['finetune_norm']:
        opt['name'] = 'finetune_{}'.format(opt['name'])
    else:
        opt['name'] = '{}_{}'.format(opt['phase'], opt['name'])

    ''' set log directory '''
    if args.output_dir_name is not None:
        experiments_root = os.path.join(opt['path']['base_dir'], args.output_dir_name)
    else:
        experiments_root = os.path.join(opt['path']['base_dir'], '{}_{}'.format(opt['name'], get_timestamp()))
    mkdirs(experiments_root)

    ''' save json '''
    write_json(opt, '{}/config.json'.format(experiments_root))

    ''' change folder relative hierarchy '''
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'base' not in key and 'root' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    ''' debug mode '''
    if 'debug' in opt['name']:
        opt['train'].update(opt['debug'])

    ''' code backup ''' 
    # 获取mom目录的路径
    mom_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 需要备份的目录列表
    backup_dirs = ['config', 'models', 'core', 'slurm', 'data', 'mom_models']
    
    # 备份目录
    for dir_name in backup_dirs:
        src_dir = os.path.join(mom_dir, dir_name)
        if os.path.exists(src_dir):
            print('Copy code from {} to {}'.format(src_dir, opt['path']['code']))
            dst_dir = os.path.join(opt['path']['code'], dir_name)
            if os.path.lexists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
    
    # 备份根目录下的Python文件
    for name in os.listdir(mom_dir):
        if name.endswith('.py') or name.endswith('.sh'):
            src_file = os.path.join(mom_dir, name)
            dst_file = os.path.join(opt['path']['code'], name)
            shutil.copy2(src_file, dst_file)
    
    ''' output visualization'''
    if args.use_color_map is not None:
        opt['use_color_map'] = args.use_color_map
    
    if args.out_type is not None:
        opt['out_type'] = args.out_type
    
    if args.scale_factor is not None:
        opt['scale_factor'] = args.scale_factor

    return dict_to_nonedict(opt)





