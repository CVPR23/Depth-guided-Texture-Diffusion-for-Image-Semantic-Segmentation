# import warnings

# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
# from mmcv.utils import Registry

# MODELS = Registry("models", parent=MMCV_MODELS)
# # ATTENTION = Registry("attention", parent=MMCV_ATTENTION)


# # BACKBONES = MODELS
# # NECKS = MODELS
# # HEADS = MODELS
# # LOSSES = MODELS
# DEPTHER = MODELS


# # def build_backbone(cfg):
# #     """Build backbone."""
# #     return BACKBONES.build(cfg)


# # def build_neck(cfg):
# #     """Build neck."""
# #     return NECKS.build(cfg)


# # def build_head(cfg):
# #     """Build head."""
# #     return HEADS.build(cfg)


# # def build_loss(cfg):
# #     """Build loss."""
# #     return LOSSES.build(cfg)


# def build_depther(cfg, train_cfg=None, test_cfg=None):
#     """Build depther."""
#     if train_cfg is not None or test_cfg is not None:
#         warnings.warn("train_cfg and test_cfg is deprecated, " "please specify them in model", UserWarning)
#     assert cfg.get("train_cfg") is None or train_cfg is None, "train_cfg specified in both outer field and model field "
#     assert cfg.get("test_cfg") is None or test_cfg is None, "test_cfg specified in both outer field and model field "
#     return DEPTHER.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F

from dinov2.eval.depth.models import build_depther


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load('../pretrain/dinov2', 'dinov2_vitl14', source='local') # sota depth estimation#
# backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()

import urllib

import mmcv
from mmcv.runner import load_checkpoint


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_DATASET = "nyu" # in ("nyu", "kitti")
HEAD_TYPE = "linear" # in ("linear", "linear4", "dpt")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = create_depther(
    cfg,
    backbone_model=backbone_model,
    backbone_size=BACKBONE_SIZE,
    head_type=HEAD_TYPE,
)

load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.eval()
model.cuda()

import urllib

from PIL import Image


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


image = load_image_from_url(EXAMPLE_IMAGE_URL)

import os
from tqdm import tqdm
path = '/root/autodl-tmp/dataset/cod_train/Imgs/' #TestDataset/STERE/RGB
save_path = '/root/autodl-tmp/dataset/cod_train/Depth_nyu_linear_large_1x/' #TestDataset/STERE
if not os.path.exists(save_path):
    os.mkdir(save_path)
i = 0
for filename in tqdm(os.listdir(path)):
    i+=1
    if i>3286:
        if (filename.endswith(".jpg") or filename.endswith(".png")):  # 假设只处理.jpg格式的图片文件
            image_path = os.path.join(path, filename)  # 图片文件的完整路径

            # 打开图片文件
            image = Image.open(image_path).convert("RGB")

            # 处理图片
            import matplotlib
            from torchvision import transforms
            def make_depth_transform() -> transforms.Compose:
                return transforms.Compose([
                    transforms.ToTensor(),
                    lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
                    transforms.Normalize(
                        mean=(123.675, 116.28, 103.53),
                        std=(58.395, 57.12, 57.375),
                    ),
                ])
            def render_depth(values, colormap_name="magma_r") -> Image:
                min_value, max_value = values.min(), values.max()
                normalized_values = (values - min_value) / (max_value - min_value)

                colormap = matplotlib.colormaps[colormap_name]
                colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
                colors = colors[:, :, :3] # Discard alpha component
                return Image.fromarray(colors)
            transform = make_depth_transform()

            # max_size = 1980
            # ratio = min(max_size / image.width, max_size / image.width)
        #     try:
        #         ratio = 1.5
        #         scale_factor = ratio
        #         rescaled_image = image.resize((int(scale_factor * image.width), int(scale_factor * image.height)))#
        #         transformed_image = transform(rescaled_image)
        #         batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image
        #         with torch.inference_mode():
        #             result = model.whole_inference(batch, img_meta=None, rescale=True)
        #         depth_image = render_depth(result.squeeze().cpu())
        #     except:
        #         try:
        #             scale_factor = 1
        #             rescaled_image = image.resize((int(scale_factor * image.width), int(scale_factor * image.height)))#
        #             transformed_image = transform(rescaled_image)
        #             batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image
        #             with torch.inference_mode():
        #                 result = model.whole_inference(batch, img_meta=None, rescale=True)
        #             depth_image = render_depth(result.squeeze().cpu())
        #         except:
        #             try:
        #                 scale_factor = 3/4
        #                 rescaled_image = image.resize((int(scale_factor * image.width), int(scale_factor * image.height)))#
        #                 transformed_image = transform(rescaled_image)
        #                 batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image
        #                 with torch.inference_mode():
        #                     result = model.whole_inference(batch, img_meta=None, rescale=True)
        #                 depth_image = render_depth(result.squeeze().cpu())
        #             except:
        #                 scale_factor = 0.5
        #                 rescaled_image = image.resize((int(scale_factor * image.width), int(scale_factor * image.height)))#
        #                 transformed_image = transform(rescaled_image)
        #                 batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image
        #                 with torch.inference_mode():
        #                     result = model.whole_inference(batch, img_meta=None, rescale=True)
        #                 depth_image = render_depth(result.squeeze().cpu())

        scale_factor = 1
        rescaled_image = image.resize((int(scale_factor * image.width), int(scale_factor * image.height)))#
        transformed_image = transform(rescaled_image)
        batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image
        with torch.inference_mode():
            result = model.whole_inference(batch, img_meta=None, rescale=True)
        depth_image = render_depth(result.squeeze().cpu())


        # 在文件名后添加"_depth"后缀
        new_filename = os.path.splitext(filename)[0] + "_depth" + os.path.splitext(filename)[1]

        # 保存图片
        depth_image.save(os.path.join(save_path, new_filename))
