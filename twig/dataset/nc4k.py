from torch.utils.data import Dataset
from nest import export 
import torchvision.transforms as transforms
import os
from typing import Union, Optional
from PIL import Image
import torch

@export
class NC4K(Dataset):
    """Load data for COD testing on NC4K"""

    def __init__(self, data_dir: str, split: str, image_size: Optional[Union[tuple, list]] = None):
        self.trainsize = 384
        if split == 'train':
            raise ValueError(f'The NC4K is usually used for testing') 
        elif split == 'test' or split == 'val':
            self.images = [os.path.join(data_dir, 'train', 'Image', f) for f in os.listdir(os.path.join(data_dir, 'train', 'Image'))]
            self.gts = [os.path.join(data_dir, 'train', 'GT', f) for f in os.listdir(os.path.join(data_dir, 'train', 'GT'))]            
        else:
            raise NotImplementedError(f'Unsupported split {split}')           
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()

        self.raw_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        raw = Image.open(self.images[index])
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        raw = self.raw_transform(raw)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return {
            'raw': raw,
            'input': image, 
            'label': gt
        }

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