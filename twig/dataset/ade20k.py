
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from nest import export
import cv2
from torchvision import transforms
import torch

CLASSES = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
    'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
    'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
    'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag')

PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

@export
class ADE20K(Dataset):
    """Load data for segmentation on ADE20K"""
    
    def __init__(self, data_dir: str, split: str='train'):
        self.trainsize=384
        if split == 'train':
            self.images = [os.path.join(data_dir, 'images', 'training', f) for f in os.listdir(os.path.join(data_dir, 'images', 'training'))]
            self.gts = [os.path.join(data_dir, 'annotations', 'training', f) for f in os.listdir(os.path.join(data_dir, 'annotations', 'training'))]
        elif split == 'val':
            self.images = [os.path.join(data_dir, 'images', 'validation', f) for f in os.listdir(os.path.join(data_dir, 'images', 'validation'))]
            self.gts = [os.path.join(data_dir, 'annotations', 'validation', f) for f in os.listdir(os.path.join(data_dir, 'annotations', 'validation'))]         
        elif split == 'test':
            self.images = [os.path.join(data_dir, 'images', 'testing', f) for f in os.listdir(os.path.join(data_dir, 'images', 'testing'))]
            self.gts = [os.path.join(data_dir, 'annotations', 'testing', f) for f in os.listdir(os.path.join(data_dir, 'annotations', 'testing'))]   
        else:
            raise NotImplementedError(f'Unsupported split {split}')                     
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # if split == 'val':
        #     self.images = self.images[:100]
        #     self.gts = self.gts[:100]
        self.cats = CLASSES
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

    def __len__(self):
        return len(self.images)
    
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
            'sam_input': torch.tensor([0]),
            'label': gt
        }




    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            gt = np.array(Image.open(f), dtype=np.uint8)
            gt[gt==0] = 255
            gt = gt - 1
            gt[gt==254] = 255
            gt = Image.fromarray(gt)
            return gt

    # def load_image(self, idx):
    #     image = cv2.imdecode(
    #         np.fromfile(self.imagepath % self.ids[idx], dtype=np.uint8),
    #         cv2.IMREAD_COLOR)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     return image.astype(np.float32)

    # def load_mask(self, idx):
    #     # h,w
    #     mask = np.array(Image.open(self.maskpath % self.ids[idx]),
    #                     dtype=np.uint8)
    #     # If class 0 is the background class and you want to ignore it when calculating the evaluation index,
    #     # you need to set reduce_zero_label=True.
    #     if self.reduce_zero_label:
    #         # avoid using underflow conversion
    #         mask[mask == 0] = 255
    #         mask = mask - 1
    #         # background class 0 transform to class 255,class 1~150 transform to 0~149
    #         mask[mask == 254] = 255

    #     return mask.astype(np.float32)




# from torch.utils.data import Dataset
# import os.path as osp
# import mmcv
# import numpy as np
# from PIL import Image
# from nest import export

        
# class CustomDataset(Dataset):
#     """Custom dataset for semantic segmentation. An example of file structure
#     is as followed.
#     .. code-block:: none
#         ├── data
#         │   ├── my_dataset
#         │   │   ├── img_dir
#         │   │   │   ├── train
#         │   │   │   │   ├── xxx{img_suffix}
#         │   │   │   │   ├── yyy{img_suffix}
#         │   │   │   │   ├── zzz{img_suffix}
#         │   │   │   ├── val
#         │   │   ├── ann_dir
#         │   │   │   ├── train
#         │   │   │   │   ├── xxx{seg_map_suffix}
#         │   │   │   │   ├── yyy{seg_map_suffix}
#         │   │   │   │   ├── zzz{seg_map_suffix}
#         │   │   │   ├── val
#     The img/gt_semantic_seg pair of CustomDataset should be of the same
#     except suffix. A valid img/gt_semantic_seg filename pair should be like
#     ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
#     in the suffix). If split is given, then ``xxx`` is specified in txt file.
#     Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
#     Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.
#     Args:
#         pipeline (list[dict]): Processing pipeline
#         img_dir (str): Path to image directory
#         img_suffix (str): Suffix of images. Default: '.jpg'
#         ann_dir (str, optional): Path to annotation directory. Default: None
#         seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
#         split (str, optional): Split txt file. If split is specified, only
#             file with suffix in the splits will be loaded. Otherwise, all
#             images in img_dir/ann_dir will be loaded. Default: None
#         data_root (str, optional): Data root for img_dir/ann_dir. Default:
#             None.
#         test_mode (bool): If test_mode=True, gt wouldn't be loaded.
#         ignore_index (int): The label index to be ignored. Default: 255
#         reduce_zero_label (bool): Whether to mark label zero as ignored.
#             Default: False
#         classes (str | Sequence[str], optional): Specify classes to load.
#             If is None, ``cls.CLASSES`` will be used. Default: None.
#         palette (Sequence[Sequence[int]]] | np.ndarray | None):
#             The palette of segmentation map. If None is given, and
#             self.PALETTE is None, random palette will be generated.
#             Default: None
#         gt_seg_map_loader_cfg (dict): build LoadAnnotations to load gt for
#             evaluation, load from disk by default. Default: ``dict()``.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#     """

#     CLASSES = None

#     PALETTE = None

#     def __init__(self,
#                  pipeline,
#                  img_dir,
#                  img_suffix='.jpg',
#                  ann_dir=None,
#                  seg_map_suffix='.png',
#                  split=None,
#                  data_root=None,
#                  test_mode=False,
#                  ignore_index=255,
#                  reduce_zero_label=False,
#                  classes=None,
#                  palette=None,
#                  gt_seg_map_loader_cfg=dict(),
#                  file_client_args=dict(backend='disk')):
#         self.img_dir = img_dir
#         self.img_suffix = img_suffix
#         self.ann_dir = ann_dir
#         self.seg_map_suffix = seg_map_suffix
#         self.split = split
#         self.data_root = data_root
#         self.test_mode = test_mode
#         self.ignore_index = ignore_index
#         self.reduce_zero_label = reduce_zero_label
#         self.label_map = None
#         self.CLASSES, self.PALETTE = self.get_classes_and_palette(
#             classes, palette)
#         self.gt_seg_map_loader = LoadAnnotations(
#             reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)

#         self.file_client_args = file_client_args
#         self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

#         if test_mode:
#             assert self.CLASSES is not None, \
#                 '`cls.CLASSES` or `classes` should be specified when testing'

#         # join paths if data_root is specified
#         if self.data_root is not None:
#             if not osp.isabs(self.img_dir):
#                 self.img_dir = osp.join(self.data_root, self.img_dir)
#             if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
#                 self.ann_dir = osp.join(self.data_root, self.ann_dir)
#             if not (self.split is None or osp.isabs(self.split)):
#                 self.split = osp.join(self.data_root, self.split)

#         # load annotations
#         self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
#                                                self.ann_dir,
#                                                self.seg_map_suffix, self.split)

#     def __len__(self):
#         """Total number of samples of data."""
#         return len(self.img_infos)

#     def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
#                          split):
#         """Load annotation from directory.
#         Args:
#             img_dir (str): Path to image directory
#             img_suffix (str): Suffix of images.
#             ann_dir (str|None): Path to annotation directory.
#             seg_map_suffix (str|None): Suffix of segmentation maps.
#             split (str|None): Split txt file. If split is specified, only file
#                 with suffix in the splits will be loaded. Otherwise, all images
#                 in img_dir/ann_dir will be loaded. Default: None
#         Returns:
#             list[dict]: All image info of dataset.
#         """

#         img_infos = []
#         if split is not None:
#             lines = mmcv.list_from_file(
#                 split, file_client_args=self.file_client_args)
#             for line in lines:
#                 img_name = line.strip()
#                 img_info = dict(filename=img_name + img_suffix)
#                 if ann_dir is not None:
#                     seg_map = img_name + seg_map_suffix
#                     img_info['ann'] = dict(seg_map=seg_map)
#                 img_infos.append(img_info)
#         else:
#             for img in self.file_client.list_dir_or_file(
#                     dir_path=img_dir,
#                     list_dir=False,
#                     suffix=img_suffix,
#                     recursive=True):
#                 img_info = dict(filename=img)
#                 if ann_dir is not None:
#                     seg_map = img.replace(img_suffix, seg_map_suffix)
#                     img_info['ann'] = dict(seg_map=seg_map)
#                 img_infos.append(img_info)
#             img_infos = sorted(img_infos, key=lambda x: x['filename'])

#         return img_infos

#     def get_ann_info(self, idx):
#         """Get annotation by index.
#         Args:
#             idx (int): Index of data.
#         Returns:
#             dict: Annotation info of specified index.
#         """

#         return self.img_infos[idx]['ann']

#     def pre_pipeline(self, results):
#         """Prepare results dict for pipeline."""
#         results['seg_fields'] = []
#         results['img_prefix'] = self.img_dir
#         results['seg_prefix'] = self.ann_dir
#         if self.custom_classes:
#             results['label_map'] = self.label_map

#     def __getitem__(self, idx):
#         """Get training/test data after pipeline.
#         Args:
#             idx (int): Index of data.
#         Returns:
#             dict: Training/test data (with annotation if `test_mode` is set
#                 False).
#         """

#         if self.test_mode:
#             return self.prepare_test_img(idx)
#         else:
#             return self.prepare_train_img(idx)

#     def prepare_train_img(self, idx):
#         """Get training data and annotations after pipeline.
#         Args:
#             idx (int): Index of data.
#         Returns:
#             dict: Training data and annotation after pipeline with new keys
#                 introduced by pipeline.
#         """

#         img_info = self.img_infos[idx]
#         ann_info = self.get_ann_info(idx)
#         results = dict(img_info=img_info, ann_info=ann_info)
#         self.pre_pipeline(results)
#         return self.pipeline(results)

#     def prepare_test_img(self, idx):
#         """Get testing data after pipeline.
#         Args:
#             idx (int): Index of data.
#         Returns:
#             dict: Testing data after pipeline with new keys introduced by
#                 pipeline.
#         """

#         img_info = self.img_infos[idx]
#         results = dict(img_info=img_info)
#         self.pre_pipeline(results)
#         return self.pipeline(results)

#     def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
#         """Place holder to format result to dataset specific output."""
#         raise NotImplementedError

#     def get_gt_seg_map_by_idx(self, index):
#         """Get one ground truth segmentation map for evaluation."""
#         ann_info = self.get_ann_info(index)
#         results = dict(ann_info=ann_info)
#         self.pre_pipeline(results)
#         self.gt_seg_map_loader(results)
#         return results['gt_semantic_seg']

#     def get_gt_seg_maps(self):
#         """Get ground truth segmentation maps for evaluation."""

#         for idx in range(len(self)):
#             ann_info = self.get_ann_info(idx)
#             results = dict(ann_info=ann_info)
#             self.pre_pipeline(results)
#             self.gt_seg_map_loader(results)
#             yield results['gt_semantic_seg']

# class LoadAnnotations(object):
#     """Load annotations for semantic segmentation.
#     Args:
#         reduce_zero_label (bool): Whether reduce all label value by 1.
#             Usually used for datasets where 0 is background label.
#             Default: False.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """

#     def __init__(self,
#                  reduce_zero_label=False,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.reduce_zero_label = reduce_zero_label
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, results):
#         """Call function to load multiple types annotations.
#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.
#         Returns:
#             dict: The dict contains loaded semantic segmentation annotations.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results.get('seg_prefix', None) is not None:
#             filename = osp.join(results['seg_prefix'],
#                                 results['ann_info']['seg_map'])
#         else:
#             filename = results['ann_info']['seg_map']
#         img_bytes = self.file_client.get(filename)
#         gt_semantic_seg = mmcv.imfrombytes(
#             img_bytes, flag='unchanged',
#             backend=self.imdecode_backend).squeeze().astype(np.uint8)
#         # reduce zero_label
#         if self.reduce_zero_label:
#             # avoid using underflow conversion
#             gt_semantic_seg[gt_semantic_seg == 0] = 255
#             gt_semantic_seg = gt_semantic_seg - 1
#             gt_semantic_seg[gt_semantic_seg == 254] = 255
#         # modify if custom classes
#         if results.get('label_map', None) is not None:
#             # Add deep copy to solve bug of repeatedly
#             # replace `gt_semantic_seg`, which is reported in
#             # https://github.com/open-mmlab/mmsegmentation/pull/1445/
#             gt_semantic_seg_copy = gt_semantic_seg.copy()
#             for old_id, new_id in results['label_map'].items():
#                 gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
#         results['gt_semantic_seg'] = gt_semantic_seg
#         results['seg_fields'].append('gt_semantic_seg')
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return 

# @export
# class ADE20KDataset(CustomDataset):
#     """ADE20K dataset.
#     In segmentation map annotation for ADE20K, 0 stands for background, which
#     is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
#     The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
#     '.png'.
#     """
#     CLASSES = (
#         'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
#         'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
#         'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
#         'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
#         'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
#         'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
#         'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
#         'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
#         'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
#         'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
#         'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
#         'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
#         'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
#         'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
#         'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
#         'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
#         'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
#         'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
#         'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
#         'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
#         'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
#         'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
#         'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
#         'clock', 'flag')

#     PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
#                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
#                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
#                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
#                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
#                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
#                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
#                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
#                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
#                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
#                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
#                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
#                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
#                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
#                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
#                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
#                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
#                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
#                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
#                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
#                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
#                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
#                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
#                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
#                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
#                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
#                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
#                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
#                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
#                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
#                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
#                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
#                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
#                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
#                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
#                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
#                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
#                [102, 255, 0], [92, 0, 255]]

#     def __init__(self, **kwargs):
#         super(ADE20KDataset, self).__init__(
#             img_suffix='.jpg',
#             seg_map_suffix='.png',
#             reduce_zero_label=True,
#             **kwargs)

#     def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
#         """Write the segmentation results to images.
#         Args:
#             results (list[ndarray]): Testing results of the
#                 dataset.
#             imgfile_prefix (str): The filename prefix of the png files.
#                 If the prefix is "somepath/xxx",
#                 the png files will be named "somepath/xxx.png".
#             to_label_id (bool): whether convert output to label_id for
#                 submission.
#             indices (list[int], optional): Indices of input results, if not
#                 set, all the indices of the dataset will be used.
#                 Default: None.
#         Returns:
#             list[str: str]: result txt files which contains corresponding
#             semantic segmentation images.
#         """
#         if indices is None:
#             indices = list(range(len(self)))

#         mmcv.mkdir_or_exist(imgfile_prefix)
#         result_files = []
#         for result, idx in zip(results, indices):

#             filename = self.img_infos[idx]['filename']
#             basename = osp.splitext(osp.basename(filename))[0]

#             png_filename = osp.join(imgfile_prefix, f'{basename}.png')

#             # The  index range of official requirement is from 0 to 150.
#             # But the index range of output is from 0 to 149.
#             # That is because we set reduce_zero_label=True.
#             result = result + 1

#             output = Image.fromarray(result.astype(np.uint8))
#             output.save(png_filename)
#             result_files.append(png_filename)

#         return result_files

#     def format_results(self,
#                        results,
#                        imgfile_prefix,
#                        to_label_id=True,
#                        indices=None):
#         """Format the results into dir (standard format for ade20k evaluation).
#         Args:
#             results (list): Testing results of the dataset.
#             imgfile_prefix (str | None): The prefix of images files. It
#                 includes the file path and the prefix of filename, e.g.,
#                 "a/b/prefix".
#             to_label_id (bool): whether convert output to label_id for
#                 submission. Default: False
#             indices (list[int], optional): Indices of input results, if not
#                 set, all the indices of the dataset will be used.
#                 Default: None.
#         Returns:
#             tuple: (result_files, tmp_dir), result_files is a list containing
#                the image paths, tmp_dir is the temporal directory created
#                 for saving json/png files when img_prefix is not specified.
#         """

#         if indices is None:
#             indices = list(range(len(self)))

#         assert isinstance(results, list), 'results must be a list.'
#         assert isinstance(indices, list), 'indices must be a list.'

#         result_files = self.results2img(results, imgfile_prefix, to_label_id,
#                                         indices)
#         return 