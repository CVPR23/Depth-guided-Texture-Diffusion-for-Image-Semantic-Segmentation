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
class NYU(Dataset):
    """Load data for depth estimation on NYU"""

    def __init__(self, data_dir: str, split: str, image_size: Optional[Union[tuple, list]] = None):
        self.trainsize = 384
        self.cropsize = 224
        if split == 'train':
            self.images = [os.path.join(data_dir, 'images', f) for f in os.listdir(os.path.join(data_dir, 'images'))]
            self.gts = [os.path.join(data_dir, 'depth', f) for f in os.listdir(os.path.join(data_dir, 'depth'))]
            self.depth = [os.path.join(data_dir, 'depth', f) for f in os.listdir(os.path.join(data_dir, 'depth'))]
        elif split == 'test' or split == 'val':
            self.images = [os.path.join(data_dir, 'images', f) for f in os.listdir(os.path.join(data_dir, 'images'))]
            self.gts = [os.path.join(data_dir, 'depth', f) for f in os.listdir(os.path.join(data_dir, 'depth'))]
            self.depth = [os.path.join(data_dir, 'depth', f) for f in os.listdir(os.path.join(data_dir, 'depth'))]        
        else:
            raise NotImplementedError(f'Unsupported split {split}')                     
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depth = sorted(self.depth)
        self.filter_files()

        self.img_transform = transforms.Compose([
            # transforms.RandomApply([transforms.RandomRotation(degrees=2.5)], p=0.5),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(
            #                             brightness=[0.75, 1.25],
            #                             contrast=[0.9, 1.1],
            #                             saturation=[0.9, 1.1],
            #                             hue=0
            #                         )], p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([127.5, 127.5, 127.5], 
            [127.5, 127.5, 127.5])
        ])
        self.gt_transform = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(degrees=2.5)], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.raw_transform = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(degrees=2.5)], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
        ])
        self.depth_transform = transforms.Compose([
            # transforms.RandomApply([transforms.RandomRotation(degrees=2.5)], p=0.5),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(
            #                 brightness=[0.75, 1.25],
            #                 contrast=[0.9, 1.1],
            #                 saturation=[0.9, 1.1],
            #                 hue=0
            #             )], p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) 
        image = self.img_transform(image)
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) 
        gt = self.gt_transform(gt)
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) 
        depth = self.depth_transform(self.binary_loader(self.depth[index]))
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) 
        raw = self.raw_transform(Image.open(self.images[index]))
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

