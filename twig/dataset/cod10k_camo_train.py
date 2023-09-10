from torch.utils.data import Dataset
from nest import export 
import os
from typing import Union, Optional
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import torch

@export
class COD10K_CAMO_TRAIN(Dataset):
    """Load data for COD training on training set of COD10K and CAMO"""

    def __init__(self, data_dir: str, split: str, image_size: Optional[Union[tuple, list]] = None):
        self.trainsize = 704#384
        self.cropsize = 224
        if split == 'train':
            self.images = [os.path.join(data_dir, 'Imgs', f) for f in os.listdir(os.path.join(data_dir, 'Imgs'))]
            self.gts = [os.path.join(data_dir, 'GT', f) for f in os.listdir(os.path.join(data_dir, 'GT'))]
            # depth_dir = '/root/autodl-tmp/sw/workspace/dqnet-depth-nest/Metric3D-main/show_dirs/convlarge.0.3_150/20230901_154136/vis/cod_train'
            self.depth = [os.path.join(data_dir, 'Depth_kitti_linear_large', f) for f in os.listdir(os.path.join(data_dir, 'Depth_kitti_linear_large'))]
        elif split == 'test' or split == 'val':
            raise ValueError(f'The training set of COD10K and CAMO is usually used for training')         
        else:
            raise NotImplementedError(f'Unsupported split {split}')                     
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depth = sorted(self.depth)
        self.filter_files()

        self.img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        self.raw_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
        ])
        self.depth_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        # image = Image.open(self.images[index]).convert('RGB')
        # gt = Image.open(self.gts[index]).convert('L')
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # image, gt = self.randomcrop(np.array(image), np.array(gt))
        # image, gt = self.randomflip(np.array(image), np.array(gt))
        # image = self.img_transform(Image.fromarray(image))
        # gt = self.gt_transform(Image.fromarray(gt))
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        image = self.img_transform(image)
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        gt = self.gt_transform(gt)
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        raw = self.raw_transform(Image.open(self.images[index]))
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        depth = self.gt_transform(self.binary_loader(self.depth[index]))
        return {
            'raw': raw,
            'input': image, 
            'label': gt,
            'depth': depth
        }

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')
        
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts


    def __len__(self):
        return len(self.images)

class RandomCrop(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3] 

class RandomFlip(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1,:].copy(), mask[:, ::-1].copy() 
        else:
            if mask is None:
                return image
            return image, mask 