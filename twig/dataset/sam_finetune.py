from torch.utils.data import Dataset
from nest import export 
import os
from typing import Union, Optional
from PIL import Image
from torchvision import transforms
# Preprocess the images
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

@export
class COD_FINETUNING(Dataset):
    """Load data for fintuning SAM on COD"""

    def __init__(self, data_dir: str, split: str, image_size: Optional[Union[tuple, list]] = None):
        self.trainsize = 384
        if split == 'train':
            self.images = [os.path.join(data_dir, 'TrainDataset', 'Imgs', f) for f in os.listdir(os.path.join(data_dir, 'TrainDataset', 'Imgs'))]
            self.gts = [os.path.join(data_dir, 'TrainDataset', 'GT', f) for f in os.listdir(os.path.join(data_dir, 'TrainDataset', 'GT'))]
        
        elif split == 'test' or split == 'val':
            raise ValueError(f'The training set of COD10K and CAMO is usually used for training')         
        else:
            raise NotImplementedError(f'Unsupported split {split}')                     
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()

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

        self.raw_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
        ])
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def __getitem__(self, index):
        dqnet_input = Image.open(self.images[index]).convert('RGB')
        # gt = Image.open(self.gts[index]).convert('L')
        raw = Image.open(self.images[index])
        raw = self.raw_transform(raw)
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.trainsize,self.trainsize))
        gt = cv2.imread(self.gts[index])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt = cv2.resize(gt, (self.trainsize,self.trainsize))
        dqnet_input = self.img_transform(dqnet_input)
        # gt = self.gt_transform(gt)

        # get bbox with max area
        contours, hierarchy = cv2.findContours(np.array(gt),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        max_index = areas.index(max(areas))
        max_contour = contours[max_index]
        x,y,w,h = cv2.boundingRect(max_contour)
        bbox_coords = np.array([x, y, x + w, y + h])  

        # # visualize bbox
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # self.show_box(bbox_coords, plt.gca())
        # self.show_mask(gt, plt.gca())
        # plt.axis('off')
        # plt.savefig(f'visualize/bbox/{index}_bbox.png')

        # convert to format sam expects
        transform = ResizeLongestSide(1024)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        return {
            'raw': raw,#
            'input': dqnet_input,
            'sam_input': transformed_image, #dqnet_input,#
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