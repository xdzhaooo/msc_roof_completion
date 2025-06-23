
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

from .util.mask import (get_multi_gauss_mask, get_down_res_mask)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir): #读取文件夹中的所有图片,返回list
    if os.path.isfile(dir): #
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')] # read file from .flist as list
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

def lmdb_loader(path):
    pass

class RoofPatchDataset(data.Dataset): # data.Dataset is a class for loading,
    def __init__(
            self, 
            corrupted_root: str,
            data_root: str, 
            footprint_root: str,
            first_output_root: str,
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
            patch_size: List[int] = [128, 128]
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
            #transforms.Resize((image_size[0], image_size[1])),
            self.height_normalize
        ])

        self.resize = transforms.Compose([    #调整为image_size大小
            #transforms.Resize((image_size[0], image_size[1])),
        ])
        #this normalize is used for recovering corrupted image real height
        self.height_normalize2 = HeightNormalize()
        self.tfs2 = transforms.Compose([    #调整为image_size大小，再进行normalize
            #transforms.Resize((image_size[0], image_size[1])),
            self.height_normalize2
        ])



    def _process_item(self, index: int): #调用__getitem__方法时，会调用这个方法
        """Get the augmented data.

        Args:
            index (int): The index of the augmented data.

        Returns:
            List: all the index augmented data.
        """
        
        # # 使用二分查找找到图像索引
        # img_idx = bisect.bisect_right(self.cumulative_patches, index) #返回第一个大于index的索引
        # if img_idx > 0:
        #     index -= self.cumulative_patches[img_idx - 1] #index被更新为在当前图片中的index

        # raw_idx = img_idx
        # Get the index of object before augmentation
        raw_idx = int(index // (len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat)) 
        #图片的index，每个图片会生成len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat个增强版本，所以用地板除，取整数部分
        
        footprint_path = self.footprints[raw_idx]
        footprint = self.resize(self.loader(footprint_path))
        footprint = torch.gt(footprint, 0).float() #greater than，大于0的为1，小于0的为0
        
        img_path = self.imgs[raw_idx]  #gt
        img_raw = torch.zeros_like(footprint) if self.no_height else self.loader(img_path) #如果no_height为True，img_raw为全0，否则为img_path对应的图片
        img = self.tfs(img_raw) #范围是【-1，1】


        corrupted_img_path = self.corrupted_img[raw_idx]  #corrupted
        corrupted_img = torch.zeros_like(footprint) if self.no_height else self.loader(corrupted_img_path)#如果no_height为True，corrupted_img为全0，否则为corrupted_img_path对应的图片
        #corrupted_img = self.tfs2(corrupted_img) #范围是【-1，1】
        first_output_path = self.first_output[raw_idx]  #第一次diffusion的输出结果
        first_output = torch.zeros_like(footprint) if self.no_height else self.loader(first_output_path)#如果no_height为True，first_output为全0，否则为first_output_path对应的图片
        first_output = transforms.Resize(corrupted_img.size()[-2:])(first_output) #调整为corrupted_img的大小

        #将tfs1的值付给tfs2
        # self.height_normalize2.mid_height = self.height_normalize.mid_height
        # self.height_normalize2.height_range = self.height_normalize.height_range
        # self.height_normalize2.valid_mask = self.height_normalize.valid_mask
        # corrupted_img = self.tfs2.transforms[0](corrupted_img) #调整为image_size大小
        def normalize(img, mid_height, height_range, valid_mask):
            img -= mid_height
            img /= height_range * 2
            img[~valid_mask] = -1
            return img
        corrupted_img = normalize(corrupted_img, self.height_normalize.mid_height, self.height_normalize.height_range, self.height_normalize.valid_mask)
        # corrupted_img = self.tfs2(corrupted_img) #范围是【-1，1】
        first_output = normalize(first_output, self.height_normalize.mid_height, self.height_normalize.height_range, self.height_normalize.valid_mask)
        # first_output = self.tfs2(first_output) #范围是【-1，1】
        corrupted_img = torch.cat([corrupted_img, first_output], dim=0) #将corrupted_img和first_output拼接起来,维度是[2, size, size]


        
        # Rotate
        rot_idx = int(index % self.n_aug_deg) #取余数，确保在0-n_aug_deg-1,所有增强版本都会依次用到 0°、90°、180°、270，注意是依次
        rot_deg = rot_idx * self.rotate_deg

        img = F.rotate(img, rot_deg)
        footprint = F.rotate(footprint, rot_deg)
        footprint = np.where(footprint < 0, 0, footprint) #用途：将小于0的值替换为0

        corrupted_img = F.rotate(corrupted_img, rot_deg)


        from torchvision.utils import save_image
        # # Image.fromarray(((gt_img.numpy()[0] + 1) / 2 * 10 * 256).astype('uint16')).save(f'./debug/ gt_{index}.png)') #保存图片
        # # Image.fromarray(((corrupted_img.numpy()[0] + 1) / 2 * 10 * 256).astype('uint16')).save(f'./debug/ corrupted_{index}.png)') #保存图片
        # # Image.fromarray((mask.numpy()[0] * 255).astype('uint8')).save(f'./debug/ mask_{index}.png)') #保存图片
        # # Image.fromarray((footprint.numpy()[0] * 255).astype('uint8')).save(f'./debug/ footprint_{index}.png)') #保存图片
        # save_image((img + 1) / 2, f'./debug/gt_rot_{index}.png')
        # save_image((torch.tensor(corrupted_img) + 1) / 2, f'./debug/corrupted_rot_{index}.png')



        if len(self.masks) > 0:
            mask_path = self.masks[raw_idx]
            mask = self.resize(self.loader(mask_path))
            mask = mask[:1,...]
            mask = F.rotate(mask, rot_deg)
        else:
            fp = footprint if self.use_footprint else np.ones_like(footprint) #如果use_footprint为False，fp为全1，否则为footprint
            mask = self.get_mask(index, fp) #范围是【0，1】，用于删除像素. footprint 1代表有，0代表没有
        
        footprint = torch.from_numpy(footprint) #范围是【0，1】

        pad_x = self.patch_size[1] // 2  # 水平填充
        pad_y = self.patch_size[0] // 2  # 垂直填充
        
        #padding
        # Padding
        img_pad = torch.nn.functional.pad(img, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=-1)
        footprint_pad = torch.nn.functional.pad(footprint, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=0)
        corrupted_img_pad = torch.nn.functional.pad(corrupted_img, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=-1)
        mask_pad = torch.nn.functional.pad(mask, pad=(pad_x, pad_x, pad_y, pad_y), mode="constant", value=0)


        #loss mask
        loss_mask = torch.zeros_like(img_pad)
        loss_mask[:,pad_x: -pad_x, pad_y: -pad_y] = 1



        return img, img_path, mask, footprint, corrupted_img, img_pad, corrupted_img_pad, mask_pad, footprint_pad, loss_mask


    def __getitem__(self, index: int): #调用方法：dataloader[index]，会调用这个方法
        """Get the augmented data.

        Args:
            index (int): The index of augmented data.

        Returns:
            dict: The dictionary of images used by model or visualization.
        """
        ret = {}

        #gt_img, img_path, mask, footprint, corrupted_img = self._process_item(index)
        gt_img, img_path, mask, footprint, corrupted_img, gt_img_pad, corrupted_img_pad, mask_pad, footprint_pad, loss_mask = self._process_item(index)
        ori_img = gt_img
        gt_img = gt_img
        corrupted_img = corrupted_img
        mask = mask
        footprint = footprint



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


        # if index >= 40 and index<=60:
        #     from torchvision.utils import save_image
        #     #Image.fromarray(((cond_img.numpy()[0] + 1) / 2 * 10 * 256).astype('uint16')).save('./debug/' + os.path.basename(ret['path'])) #保存图片
        #     save_image((torch.tensor(img) + 1) / 2, f'./debug/building_ori{index}.png')
        #     save_image((torch.tensor(gt_img) + 1) / 2, f'./debug/building_gt{index}.png')

        # Synthesize missing pixels 
        cond_img = img[0,:,:] * (1. - mask) - mask # set the pixel with mask=1 to -1，mask 0代表保留，1代表删除（值为-1），删除一些像素
        cond_img = torch.cat([cond_img, img[1,:,:].unsqueeze(0)], dim=0)  #增加一个维度，变成[2, size, size]
            #footprint 1代表是，0代表不是



        cond_img = cond_img * footprint - (1 - footprint) # set the pixel outside of footprint as -1，这是模型输入的条件图像.缺损图像，去掉footprint外的像素。footprint中0代表不保留，1代表保留
        # if index >=40 and index<=60:
        #     from torchvision.utils import save_image
        #     #Image.fromarray(((cond_img.numpy()[0] + 1) / 2 * 10 * 256).astype('uint16')).save('./debug/' + os.path.basename(ret['path'])) #保存图片
        #     save_image((torch.tensor(cond_img) + 1) / 2, f'./debug/building_afterft{index}.png')
        #     save_image((torch.tensor(footprint) + 1) / 2, f'./debug/footprint{index}.png')

        mask_img = (img * (1. - mask) + mask) * footprint - (1 - footprint) # only for visualization

        # Image to be fed to the model (only cond image)
        ret['cond_image'] = cond_img


        if self.recover_real_height: #恢复真实高度,除了benchmark，其他地方都不会用到
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
        ret['patch_pos'] = []
        ret['img_ped_size'] = [gt_img_pad.size()[1], gt_img_pad.size()[2]]
        ret['loss_mask'] = loss_mask
        ret['patch_size'] = self.patch_size
        ret['ori_img_size'] = [gt_img.size()[1], gt_img.size()[2]] #所有的patch的大小

        return ret  #dataloader会把相同key的value收集到一个list中

    def __len__(self):
        """The len of augmented training set.

        Returns:
            int: The len of augmented training set
        """
        return len(self.imgs) * len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat
        # return self.total_patches

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

