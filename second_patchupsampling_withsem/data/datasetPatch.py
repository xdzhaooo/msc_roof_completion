from typing import List
from data.util.noise import add_gauss_noise, add_outlier_noise

from data.util.roof_augment import add_tree_noise, random_height_scaling
from data.util.roof_transform import HeightNormalize
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import os
import torch
import numpy as np
import bisect
import torch.nn.functional
import lmdb
import pickle
import time
from scipy.ndimage import zoom
import random

from .util.mask import (get_multi_gauss_mask, get_down_res_mask)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir): #读取文件夹中的所有图片,返回list
    if os.path.isfile(dir): #
        # 修正处理单行文件的bug
        try:
            raw_data = np.genfromtxt(dir, dtype=str, encoding='utf-8')
            # 如果是0维数组（单个元素），转换为列表
            if raw_data.ndim == 0:
                images = [str(raw_data)]
            else:
                images = [i for i in raw_data] # read file from .flist as list
        except:
            # 如果genfromtxt失败，手动读取
            with open(dir, 'r') as f:
                images = [line.strip() for line in f.readlines() if line.strip()]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)): # walk through the directory
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images #return list of images

def KITTI_pil_loader(path):
    depth = np.array(Image.open(path))
    return depth.astype(np.float32) / 256.0 # convert to meters 

def roof_pil_loader(path): #
    height = KITTI_pil_loader(path)
    return transforms.ToTensor()(height)

def pil_loader(path, mode='L'):
    return Image.open(path).convert(mode)

class RoofPatchDataset(data.Dataset): # data.Dataset is a class for loading,
    def __init__(
            self, 
            corrupted_root: str,
            data_root: str, 
            footprint_root: str,
            first_output_root: str, #第一次diffusion的输出结果
            footprint_as_mask: bool = False,
            mask_root: str = None, 
            noise_config: dict = {},
            mask_config: dict = {},
            data_aug: dict = {}, 
            data_len: int = -1,
            recover_real_height: bool = False,
            use_footprint: bool = True,
            no_height: bool = False,
            image_size: List[int] = [128, 128], 
            loader: callable = roof_pil_loader,
            patch_size: List[int] = [128, 128],
            lmdb_root: str = None
        ):
        """Reconstruct root height image conditioned on incomplete height image.

        Args:
            data_root (str): Path to height_map.flist.
            footprint_root (str): Path to footprint.flist.
            footprint_as_mask (bool, optional): Use footprint as the mask. Defaults to False.
            mask_root (str, optional): Path to mask.flist. Defaults to None.
            noise_config (dict, optional): Parameters for synthesizing scanning or environmental noise into height maps. Note: This is distinct from the diffusion process!
            mask_config (dict, optional): Parameters for generating incompleteness mask. Defaults to {}.
            data_aug (dict, optional): Setting for data augmentation e.g. rotation. Defaults to {}.
            data_len (int, optional): The number of data points to be trained. Defaults to -1, which uses all available data.
            recover_real_height (bool, optional): If set the true, return the realworld height (meters * 256), otherwise within range of [-1, 1].
            use_footprint (bool, optional): If set to false, entire image will be treated as footprints (images filled with only ones).
            no_height (bool, optional): Set to true to replace the height map with all zeros. 
            image_size (list, optional): Image dimensions. Defaults to [128, 128].
            loader (function, optional): Image loader. Defaults to pil_loader.
        """
        
        self.loader = loader
        self.mask_config = mask_config
        self.noise_config = noise_config
        self.image_size = image_size

        self.footprint_as_mask = footprint_as_mask
        self.use_footprint = use_footprint
        self.recover_real_height = recover_real_height
        self.no_height = no_height
        self.patch_size = patch_size


        # Sythesizing borken/incomplete images  #模拟缺失像素
        self.down_res = mask_config.get("down_res_pct", [0]) # randomly remove n % of pixel within the footprint
        self.local_remove = mask_config.get("local_remove", [[0, 0, 0]]) # locally remove pixel using gauss filter within the footprint [min_sigma, max_sigma, n_gaussian]
        self.local_remove_percentage = mask_config.get("local_remove_percentage", -1)

        # Sythesizing scan/environmental noise into height map #添加噪声
        self.min_gauss_noise_sigma = noise_config.get("min_gauss_noise_sigma", 0) # min variance of gauss noise to add to conditional image
        self.max_gauss_noise_sigma = noise_config.get("max_gauss_noise_sigma", 0) # max variance of gauss noise to add to conditional image
        self.outlier_noise_perc = noise_config.get("outlier_noise_percentage", 0) # percentage of choosing pixels and assign a random value from -1 to 1 in conditional image

        # Data augmentation
        self.repeat = data_aug.get("repeat", 1) # repeat same image. 1 means no repeat
        self.rotate_deg = data_aug.get("rotate", 360) # rotate image by n degree. n=0,360 means no rotation.
        self.rotate_deg = 360 if self.rotate_deg == 0 else self.rotate_deg
        self.n_aug_deg = int(360 // self.rotate_deg) # number of images with rotation augmentation. e.g. 90 degreee => 4 augmented images (0, 90, 180, 270)
        self.height_scale_prob = data_aug.get("height_scale_probability", 0)

        # Tree augmentation
        self.tree_aug = data_aug.get("tree", None)
        if self.tree_aug is not None:
            self.trees = make_dataset(self.tree_aug['flist_path'])

        # Functions for scaling and normalizing images
        self.height_normalize = HeightNormalize()
        self.tfs = transforms.Compose([    #调整为image_size大小，再进行normalize
            # transforms.Resize((image_size[0], image_size[1])),
            self.height_normalize
        ])

        self.resize = transforms.Compose([    #调整为image_size大小
            # transforms.Resize((image_size[0], image_size[1])),
        ])
        #this normalize is used for recovering corrupted image real height
        self.height_normalize2 = HeightNormalize()
        self.tfs2 = transforms.Compose([    #调整为image_size大小，再进行normalize,nearest
            transforms.Resize((image_size[0], image_size[1]),interpolation=transforms.InterpolationMode.NEAREST),
            self.height_normalize2
        ])

        self.lmdb_root = None
        # LMDB 数据预加载到内存
        if lmdb_root is not None:
            self.lmdb_root = lmdb_root
            self.data_memory = {}  # 存储所有数据的内存字典
            env = lmdb.open(self.lmdb_root, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                index_list = [key.decode("ascii") for key, _ in cursor]
                n_data = data_len if data_len > 0 else len(index_list)
                self.index_list = index_list[:n_data]
                
                # 预加载所有数据到内存
                for key in self.index_list:
                    img = txn.get(key.encode("ascii"))
                    image_array = pickle.loads(img)  # [H, W, 4]
                    self.data_memory[key] = image_array  # 直接存储 NumPy 数组
                print(f"Loaded {len(self.data_memory)} images into memory from LMDB, total size: {sum(arr.nbytes for arr in self.data_memory.values()) / 1024**3:.2f} GB")
            env.close()  # 加载完成后关闭环境，后续不再需要访问磁盘

            # 计算 patch 数量（保持原有逻辑）
            self.total_patches_per_img = []
            for key in self.index_list:
                image_array = self.data_memory[key]
                h, w, c = image_array.shape
                assert c == 4, f'channel should be 4, but got {c}'
                padded_size = (h + self.patch_size[0], w + self.patch_size[1])
                # 修改步长为patch_size，patch大小为2*patch_size
                step_h = self.patch_size[0]  # 128
                step_w = self.patch_size[1]  # 128
                patches_per_img = int(((padded_size[0] - 2*self.patch_size[0])//step_h + 1) * ((padded_size[1] - 2*self.patch_size[1])//step_w + 1))
                self.total_patches_per_img.append(patches_per_img * len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat)
            self.total_patches = sum(self.total_patches_per_img)
            self.cumulative_patches = [sum(self.total_patches_per_img[:i+1]) for i in range(len(self.total_patches_per_img))]
            self.masks = None
        else:
            # Load data paths from .flist file
            imgs = make_dataset(data_root) #gt
            footprints = make_dataset(footprint_root)
            masks = make_dataset(mask_root) if mask_root is not None else []
            corrupted_img = make_dataset(corrupted_root) if corrupted_root is not None else [] #缺损的图像
            first_output = make_dataset(first_output_root) if first_output_root is not None else [] #第一次diffusion的输出结果

            # Load specified number of data
            n_data = data_len if data_len > 0 else len(imgs)
            self.imgs = imgs[:n_data] 
            self.footprints = footprints[:n_data]
            self.masks = masks[:n_data]
            self.corrupted_img = corrupted_img[:n_data] 
            self.first_output = first_output[:n_data]

            #计算patch的数量
            self.total_patches_per_img = []    
            for img_path in self.imgs:
                img = self.loader(img_path)
                padded_img = F.pad(img, padding=(self.patch_size[0] // 2, self.patch_size[1] // 2, self.patch_size[0] // 2, self.patch_size[1] // 2))
                # 修改步长为patch_size，patch大小为2*patch_size
                step_h = self.patch_size[0]  # 128
                step_w = self.patch_size[1]  # 128
                patches_per_img = int(((padded_img.size(1) - 2*self.patch_size[0])//step_h + 1) * ((padded_img.size(2) - 2*self.patch_size[1])//step_w + 1))
                #patches_per_img = (padded_img.size(1) // self.patch_size[0]) * (padded_img.size(2) // self.patch_size[1])
                self.total_patches_per_img.append(patches_per_img * len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat)
            
            self.total_patches = sum(self.total_patches_per_img)
            self.cumulative_patches = [sum(self.total_patches_per_img[:i+1]) for i in range(len(self.total_patches_per_img))]#累积求和
    
    def _init_db(self):
        # 在每个 worker 中初始化 LMDB 环境
        if self.env is None:
            self.env = lmdb.open(self.lmdb_root, readonly=True, lock=False)

    def _process_item(self, index: int): #调用__getitem__方法时，会调用这个方法
        """Get the augmented data.

        Args:
            index (int): The index of the augmented data.

        Returns:
            List: all the index augmented data.
        """
        start1 = time.time()
    # ... 现有代码 ...
 
        # 使用二分查找找到图像索引
        img_idx = bisect.bisect_right(self.cumulative_patches, index) #返回第一个大于index的索引
        if img_idx > 0:
            index -= self.cumulative_patches[img_idx - 1] #index被更新为在当前图片中的index
        raw_idx = img_idx
        # Get the index of object before augmentation
        # raw_idx = int(index // (len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat)) 
        #图片的index，每个图片会生成len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat个增强版本，所以用地板除，取整数部分
    

        #lmdb processing
        if self.lmdb_root is not None:
                key = self.index_list[img_idx]
                image_array = self.data_memory[key].astype(np.float32) # 直接从内存读取
                corrupt, first_output, footprint, heightmap = np.split(image_array, 4, axis=-1)
                footprint = transforms.ToTensor()(footprint.squeeze(-1)).to(torch.float32)
                footprint = torch.gt(footprint, 0).float()
                corrupted_img = transforms.ToTensor()(corrupt.squeeze(-1)).to(torch.float32) / 256.0
                first_output = transforms.ToTensor()(first_output.squeeze(-1)).to(torch.float32) / 256.0
                img_raw = transforms.ToTensor()(heightmap.squeeze(-1)).to(torch.float32) / 256.0
                img = self.tfs(img_raw)
                img_path = key
        else:

            
            img_path = self.imgs[raw_idx]  #gt，返回的是图片的路径
            img_raw = torch.zeros_like(footprint) if self.no_height else self.loader(img_path) #如果no_height为True，img_raw为全0，否则为img_path对应的图片
            img = self.tfs(img_raw) #范围是【-1，1】


            corrupted_img_path = self.corrupted_img[raw_idx]  #corrupted
            corrupted_img = torch.zeros_like(footprint) if self.no_height else self.loader(corrupted_img_path)#如果no_height为True，corrupted_img为全0，否则为corrupted_img_path对应的图片


            #corrupted_img = self.tfs2(corrupted_img) #范围是【-1，1】
            first_output_path = self.first_output[raw_idx]  #第一次diffusion的输出结果
            first_output = torch.zeros_like(footprint) if self.no_height else self.loader(first_output_path)#如果no_height为True，first_output为全0，否则为first_output_path对应的图片
            first_output_smallsize = first_output.clone()
            #resize to corrupted image size
            first_output = transforms.Resize(corrupted_img.size()[-2:],interpolation=transforms.InterpolationMode.NEAREST)(first_output) #调整为corrupted_img的大小

            footprint_path = self.footprints[raw_idx]
            footprint = self.resize(self.loader(footprint_path))
            footprint = torch.gt(footprint, 0).float() #greater than，大于0的为1，小于0的为0
            footprint = transforms.Resize(corrupted_img.size()[-2:],interpolation=transforms.InterpolationMode.NEAREST)(footprint)
            #print('footprint.shape:',footprint.shape, "id:",raw_idx)
        #将tfs1的值付给tfs2
        # self.height_normalize2.mid_height = self.height_normalize.mid_height
        # self.height_normalize2.height_range = self.height_normalize.height_range
        # self.height_normalize2.valid_mask = self.height_normalize.valid_mask
        # corrupted_img = self.tfs2(corrupted_img) #调整为image_size大小
        def normalize(img, mid_height, height_range, valid_mask):
            mask = img==0
            img -= mid_height
            img /= height_range * 2
            #img[~valid_mask] = -1
            img[mask] = -1.
            return img
        corrupted_img = normalize(corrupted_img, self.height_normalize.mid_height, self.height_normalize.height_range, self.height_normalize.valid_mask)
        from torchvision.utils import save_image
        save_image((corrupted_img+1)/2, "corrupted_img.png")
        # corrupted_img = self.tfs2(corrupted_img) #范围是【-1，1】
        first_output = normalize(first_output, self.height_normalize.mid_height, self.height_normalize.height_range, self.height_normalize.valid_mask)
        # first_output = self.tfs2(first_output) #范围是【-1，1】
        corrupted_img = torch.cat([corrupted_img, first_output], dim=0) #将corrupted_img和first_output拼接起来,维度是[2, size, size]
        # # Rotate
        # rot_idx = int(index % self.n_aug_deg) #取余数，确保在0-n_aug_deg-1,所有增强版本都会依次用到 0°、90°、180°、270，注意是依次
        # rot_deg = rot_idx * self.rotate_deg

        # img = F.rotate(img, rot_deg)
        # footprint = F.rotate(footprint, rot_deg)
        # footprint = np.where(footprint < 0, 0, footprint) #用途：将小于0的值替换为0

        # corrupted_img = F.rotate(corrupted_img, rot_deg)
        rot_map = {0: 0, 90: 1, 180: 2, 270: 3}  # 度数到 rot90 次数的映射

        rot_idx = int(index % self.n_aug_deg)
        rot_deg = rot_idx * self.rotate_deg

        if rot_deg in rot_map:
            #print("yes")
            k = rot_map[rot_deg]
            img = torch.rot90(img, k, [1, 2])  # 沿高度和宽度维度旋转
            footprint = torch.rot90(footprint, k, [1, 2])
            footprint = np.where(footprint < 0, 0, footprint)
            corrupted_img = torch.rot90(corrupted_img, k, [1, 2])
        else:
            print('no')
            # 如果需要其他角度，保留 F.rotate（但建议调整 self.rotate_deg 为 90）
            img = F.rotate(img, rot_deg)
            footprint = F.rotate(footprint, rot_deg)
            footprint = np.where(footprint < 0, 0, footprint)
            corrupted_img = F.rotate(corrupted_img, rot_deg)


        #print(f"_process_aug time: {time.time() - start1:.4f}s")

        if self.masks is not None and len(self.masks) > 0:
            mask_path = self.masks[raw_idx]
            mask = self.resize(self.loader(mask_path))
            mask = mask[:1,...]
            mask = F.rotate(mask, rot_deg)
        else:
            fp = footprint if self.use_footprint else np.ones_like(footprint) #如果use_footprint为False，fp为全1，否则为footprint
            mask = self.get_mask_optimized(index, fp) #范围是【0，1】，用于删除像素. footprint 1代表有，0代表没有
        
        footprint = torch.from_numpy(footprint) #范围是【0，1】



        #print(f"_process_mask time: {time.time() - start1:.4f}s")

        pad_x = self.patch_size[1] // 2  # 水平填充
        pad_y = self.patch_size[0] // 2  # 垂直填充
        
        #padding
        # Padding
        img_pad = torch.nn.functional.pad(img, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=-1)
        footprint_pad = torch.nn.functional.pad(footprint, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=0)
        corrupted_img_pad = torch.nn.functional.pad(corrupted_img, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=-1)
        mask_pad = torch.nn.functional.pad(mask, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=0)
        # footprint = torch.nn.functional.pad(footprint, padding=(self.patch_size[0] // 2, self.patch_size[1] // 2))
        # corrupted_img = torch.nn.functional.pad(corrupted_img, padding=(self.patch_size[0] // 2, self.patch_size[1] // 2))
        # mask = torch.nn.functional.pad(mask, padding=(self.patch_size[0] // 2, self.patch_size[1] // 2))
        assert mask_pad.shape==img_pad.shape, f'mask_pad.shape: {mask_pad.shape}, img_pad.shape: {img_pad.shape}.footprint_pad.shape: {footprint_pad.shape}, corrupted_img_pad.shape: {corrupted_img_pad.shape}, mask_pad.shape: {mask_pad.shape}'

        #print(f"_process_PAD time: {time.time() - start1:.4f}s")
        #patch index within the image
        patch_idx = int(index // (len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat))
        # patches_per_row= img_pad.size(1) // self.patch_size[0]
        # patches_per_col = img_pad.size(2) // self.patch_size[1]
        # 修改步长为patch_size，patch大小为2*patch_size
        step_h = self.patch_size[0]  # 128
        step_w = self.patch_size[1]  # 128
        patches_per_row = int((img_pad.size(1) - 2*self.patch_size[0])//step_h + 1)
        patches_per_col = int((img_pad.size(2) - 2*self.patch_size[1])//step_w + 1)
        assert patches_per_col > 0 and patches_per_row > 0, f'patches_per_row: {patches_per_row}, patches_per_col: {patches_per_col}, img_pad size should bigger than 2*patch_size'
        patch_pos = (patch_idx // patches_per_row, patch_idx % patches_per_col)
        #print(f'patch_pos: {patch_pos}', f'patch_idx: {patch_idx}', f'patches_per_row: {patches_per_row}', f'patches_per_col: {patches_per_col}')

        #crop - 提取2*patch_size大小的patch
        img_patch = img_pad[:, patch_pos[0] * step_h: patch_pos[0] * step_h + 2 * self.patch_size[0], patch_pos[1] * step_w: patch_pos[1] * step_w + 2 * self.patch_size[1]]
        corrupted_img_patch = corrupted_img_pad[:, patch_pos[0] * step_h: patch_pos[0] * step_h + 2 * self.patch_size[0], patch_pos[1] * step_w: patch_pos[1] * step_w + 2 * self.patch_size[1]]
        mask_patch = mask_pad[:, patch_pos[0] * step_h: patch_pos[0] * step_h + 2 * self.patch_size[0], patch_pos[1] * step_w: patch_pos[1] * step_w + 2 * self.patch_size[1]]
        footprint_patch = footprint_pad[:, patch_pos[0] * step_h: patch_pos[0] * step_h + 2 * self.patch_size[0], patch_pos[1] * step_w: patch_pos[1] * step_w + 2 * self.patch_size[1]]

        #loss mask
        loss_mask = torch.zeros_like(img_pad)
        loss_mask[:,pad_x: -pad_x, pad_y: -pad_y] = 1
        loss_mask_patch = loss_mask[:,patch_pos[0] * step_h: patch_pos[0] * step_h + 2 * self.patch_size[0], patch_pos[1] * step_w: patch_pos[1] * step_w + 2 * self.patch_size[1]]


        #position of the patch in the image - 计算2x2小patch的左上角坐标
        # 大patch的起始位置
        big_patch_start_y = patch_pos[0] * step_h
        big_patch_start_x = patch_pos[1] * step_w
        
        # 计算2x2小patch的左上角坐标
        small_patch_corners = []
        for i in range(2):
            for j in range(2):
                # 小patch的左上角位置（在填充图像坐标系中）
                corner_y = big_patch_start_y + i * self.patch_size[0]
                corner_x = big_patch_start_x + j * self.patch_size[1]
                # 归一化到[0,1]
                norm_corner_y = corner_y / img_pad.size(1)
                norm_corner_x = corner_x / img_pad.size(2)
                small_patch_corners.append([norm_corner_y, norm_corner_x])
        
        # 重新组织为2x2的形状
        centers_xy = np.array(small_patch_corners).reshape(2, 2, 2)

        #print(f"_process_item time: {time.time() - start1:.4f}s")

        return img, img_path, mask, footprint, corrupted_img, img_pad, corrupted_img_pad, mask_pad, footprint_pad, img_patch, corrupted_img_patch, mask_patch, footprint_patch,loss_mask_patch, centers_xy, raw_idx,first_output_smallsize


    def __getitem__(self, index: int): #调用方法：dataloader[index]，会调用这个方法
        """Get the augmented data.

        Args:
            index (int): The index of augmented data.

        Returns:
            dict: The dictionary of images used by model or visualization.
            
        """

        start2 = time.time()

        ret = {}

        #gt_img, img_path, mask, footprint, corrupted_img = self._process_item(index)
        gt_img, img_path, mask, footprint, corrupted_img, gt_img_pad, corrupted_img_pad, mask_pad, footprint_pad, gt_img_patch, corrupted_img_patch, mask_patch, footprint_patch,loss_mask_patch, patch_pos, raw_idx,first_output_smallsize = self._process_item(index)
        ori_img = gt_img
        gt_img = gt_img_patch
        corrupted_img = corrupted_img_patch
        mask = mask_patch
        footprint = footprint_patch




        # gt_max = torch.max(gt_img)
        # corrupted_max = torch.max(corrupted_img)
        # max_diff = gt_max - corrupted_max
        # corrupted_img = corrupted_img + max_diff

        # random height scaling
        # gt_img = random_height_scaling(gt_img, footprint, self.height_scale_prob) #随机缩放高度

        #img = gt_img.clone()
        img = corrupted_img.clone()

        # Synthesize tree noise
        # if self.tree_aug is not None:
        #     img = add_tree_noise(img, footprint.numpy(), self.tree_aug, self.trees)
        
        # Get rid of footprint information
        if not self.use_footprint:
            footprint = torch.ones_like(footprint)

        # Synthesize noise #添加噪声
        # img = add_gauss_noise(img, self.min_gauss_noise_sigma, self.max_gauss_noise_sigma)
        # img = add_outlier_noise(img, self.outlier_noise_perc)
        img = add_outlier_noise(img, self.outlier_noise_perc)


        # Synthesize missing pixels 
        # cond_img = img * (1. - mask) - mask # set the pixel with mask=1 to -1，mask 0代表保留，1代表删除（值为-1），删除一些像素
        #print("img.shape:",img.shape, "mask.shape:",mask.shape, "id:",raw_idx)
        cond_img = img[0,:,:] * (1. - mask) - mask # set the pixel with mask=1 to -1，mask 0代表保留，1代表删除（值为-1），删除一些像素
        cond_img = torch.cat([cond_img, img[1,:,:].unsqueeze(0)], dim=0)  #增加一个维度，变成[2, size, size]


        cond_img = cond_img * footprint - (1 - footprint) # set the pixel outside of footprint as -1，这是模型输入的条件图像.缺损图像，去掉footprint外的像素。footprint中0代表不保留，1代表保留


        mask_img = (img * (1. - mask) + mask) * footprint - (1 - footprint) # only for visualization

        # Image to be fed to the model (only cond image)
        ret['cond_image'] = cond_img

        if self.recover_real_height: #恢复真实高度
            gt_img /= 2
            gt_img *= self.height_normalize.height_range
            gt_img += self.height_normalize.mid_height
            gt_img *= 256
            gt_img[footprint < 1] = 0
            gt_img[~self.height_normalize.valid_mask] = 0

            cond_mask = cond_img > -1
            cond_img /= 2
            cond_img *= self.height_normalize.height_range
            cond_img += self.height_normalize.mid_height
            cond_img *= 256
            cond_img[~cond_mask] = 0

        # Images for computing loss
        ret['gt_image'] = gt_img
        ret['mask'] = footprint if self.footprint_as_mask else mask #在config中设置为True，footprint作为mask

        # Images for visualization
        ret['mask_image'] = mask_img
        ret['footprint'] = footprint
        ret['first_output'] = first_output_smallsize

        # Path of original data
        ret['path'] = img_path.rsplit("/")[-1].rsplit("\\")[-1] #解释：rsplit() 方法返回从右边开始的指定长度的字符串
        if self.repeat > 1 and self.data_io is None:
            ret['path'] = ret['path'].replace('.png', '_r{}.png'.format(index % self.repeat))
        
        # Record for recovering realworld roof height
        ret['height_range'] = self.height_normalize.height_range
        ret['mid_height'] = self.height_normalize.mid_height
        ret['height_range_corrupted'] = self.height_normalize.height_range
        ret['mid_height_corrupted'] = self.height_normalize.mid_height

        # from torchvision.utils import save_image
        # Image.fromarray(((cond_img.numpy()[0] + 1) / 2 * 10 * 256).astype('uint16')).save('./debug/' + os.path.basename(ret['path'])) #保存图片
        # save_image((torch.tensor(cond_img) + 1) / 2, f'./debug/building{index}.png')

        # from torchvision.utils import save_image
        # Image.fromarray((mask.numpy()[0] * 255).astype('uint8')).save(f'./debug/MaskMask_{index}.png') 
        # save_image(torch.Tensor(footprint), f'./debug/Mask_{index}.png')

        # from torchvision.utils import save_image
        # Image.fromarray(((gt_img.numpy()[0] + 1) / 2 * 255).astype('uint8')).save(f'./debug/GT_{index}.png')
        # save_image((gt_img + 1) / 2, f'./debug/GT_{index}.png')
        
        #记录pos和图片大小
        ret['patch_pos'] = torch.tensor(patch_pos)
        ret['img_ped_size'] = [gt_img_pad.size()[1], gt_img_pad.size()[2]]
        ret['loss_mask'] = loss_mask_patch
        ret['patch_size'] = self.patch_size
        ret['ori_img_size'] = [gt_img.size()[1], gt_img.size()[2]] #2*patch_size的大小，包含2x2个小patch

        if self.lmdb_root is not None:
            ret['lmdb_root'] = self.lmdb_root
            ret['path'] = img_path
            ret['path_index'] = index

            # ... 现有代码 ...
        #print(f"_getitem time: {time.time() - start2:.4f}s",img_path,ret['img_ped_size'])

        return ret  #dataloader会把相同key的value收集到一个list中

    def __len__(self):
        """The len of augmented training set.

        Returns:
            int: The len of augmented training set
        """
        #return len(self.imgs) * len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat
        return self.total_patches

    def get_mask(self, index: int, footprint: np.array):
        """Get the sythesized mask for removing pixels.

        mask=1 denotes the pixels to be removed
        mask=0 denotes the pixels to be kept

        Args:
            index (int): The index of augmented data.
            footprint (np.array): The index of augmented data's footprint.

        Returns:
            torch.tensor: the mask with global and local removal
        """
        dr_idx = index % len(self.down_res)
        lr_idx = index % len(self.local_remove)

        n_footprint_pixels = np.sum(footprint > 0)

        for _ in range(50):
            # global
            dr_mask = get_down_res_mask(footprint, self.down_res[dr_idx])

            # local
            min_sigma_ratio, max_sigma_ratio, n_gaussian = self.local_remove[lr_idx]
            gauss_mask = get_multi_gauss_mask(
                footprint, 
                min_sigma_ratio,
                max_sigma_ratio,
                n_gauss_mask=n_gaussian,
                remove_percentage=self.local_remove_percentage
            )
            lr_mask = gauss_mask * footprint

            # sythesize the mask for global and local removal 
            mask = np.logical_or(dr_mask, lr_mask)

            if np.sum(mask == 1) < n_footprint_pixels: # avoid all the pixel are masked out
                break

        return torch.from_numpy(mask).float()
    
    def get_mask_optimized(self, index: int, footprint: np.array):
        """生成全局和本地移除的合成掩码，不追求精确匹配"""
        dr_idx = index % len(self.down_res)
        lr_idx = index % len(self.local_remove)

        n_footprint_pixels = np.sum(footprint > 0)

        # 全局移除掩码
        dr_mask = get_down_res_mask_optimized(footprint, self.down_res[dr_idx])

        # 本地移除掩码
        min_sigma_ratio, max_sigma_ratio, n_gaussian = self.local_remove[lr_idx]
        gauss_mask = get_multi_gauss_mask_optimized(
            footprint,
            min_sigma_ratio,
            max_sigma_ratio,
            n_gauss_mask=n_gaussian,
            remove_percentage=self.local_remove_percentage
        )
        lr_mask = gauss_mask * (footprint > 0)

        # 合成掩码
        mask = np.logical_or(dr_mask, lr_mask)

        # 简单检查避免全移除（最多尝试 3 次）
        for _ in range(2):  # 减少循环次数，从 50 改为 2
            if np.sum(mask) >= n_footprint_pixels:  # 如果全被移除，重新生成
                dr_mask = get_down_res_mask_optimized(footprint, self.down_res[dr_idx])
                gauss_mask = get_multi_gauss_mask_optimized(
                    footprint,
                    min_sigma_ratio,
                    max_sigma_ratio,
                    n_gauss_mask=n_gaussian,
                    remove_percentage=self.local_remove_percentage
                )
                lr_mask = gauss_mask * (footprint > 0)
                mask = np.logical_or(dr_mask, lr_mask)
            else:
                break

        return torch.from_numpy(mask).float()


def get_down_res_mask_optimized(footprint, missing_percentage):
        """生成全局移除的掩码"""
        mask = np.zeros_like(footprint, dtype=bool)
        num_pixels = np.sum(footprint > 0)
        num_pixels_to_remove = int(num_pixels * missing_percentage / 100)

        coords = np.argwhere(footprint > 0)
        if num_pixels_to_remove >= num_pixels:  # 避免移除所有像素
            num_pixels_to_remove = num_pixels - 1
        indices_to_remove = np.random.choice(coords.shape[0], num_pixels_to_remove, replace=False)
        mask[tuple(coords[indices_to_remove].T)] = 1
        return mask * (footprint > 0)

def get_multi_gauss_mask_optimized(

    footprint: np.array,
    min_sigma_ratio: float = 0.1,
    max_sigma_ratio: float = 0.3,
    n_gauss_mask: int = 5,
    remove_percentage: float = -1,
):
    """生成基于高斯混合模型的本地移除掩码（简化版本）"""
    assert min_sigma_ratio <= max_sigma_ratio
    assert footprint.ndim == 3
    h, w = footprint.shape[1:]
    mask = np.zeros((h, w), dtype=bool)
    prob_mask = np.zeros((h, w), dtype=float)

    sigma_ratio = random.uniform(min_sigma_ratio, max_sigma_ratio)
    sigma = sigma_ratio * min(h, w)

    for _ in range(n_gauss_mask):
        x = random.randint(0, w)
        y = random.randint(0, h)
        y_grid, x_grid = np.ogrid[-y:h-y, -x:w-x]
        d = np.sqrt(y_grid ** 2 + x_grid ** 2)
        gaussian_dist = np.exp(-(d**2 / (2*sigma**2)))
        prob_mask += gaussian_dist

    valid_mask = (footprint > 0)[0]
    if remove_percentage > -1:
        pixels_to_be_removed = int(np.count_nonzero(footprint) * (remove_percentage / 100.0))
        prob_mask = prob_mask / np.max(prob_mask) if np.max(prob_mask) > 0 else prob_mask
        threshold = np.percentile(prob_mask[valid_mask], 100 * (1 - pixels_to_be_removed / np.count_nonzero(valid_mask)))
        mask = (prob_mask >= threshold) & valid_mask
    else:
        rand_vals = np.random.rand(h, w)
        mask = (prob_mask > rand_vals) & valid_mask

    return mask

