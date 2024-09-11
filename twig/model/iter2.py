from __future__ import absolute_import, division, print_function
import math
from timm.models.resnet import Bottleneck
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, MMDistributedDataParallel
from nest import export
from mmengine.hooks import Hook

from transformers import AutoImageProcessor, DPTForDepthEstimation
from typing import Optional, Any
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
from segment_anything import sam_model_registry
from torchcam.methods import CAM
import matplotlib.pyplot as plt
import cv2 
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.nn.functional import threshold, normalize
import random
from timm import create_model
import os

# current_path = os.getcwd()
# print("当前路径为：" + current_path)
# from timm.models import build_model_with_cfg
# import mmseg.models.decode_heads.UPerHead as UPerHead
import mmseg


@export
class cod2(BaseModel):
    """DQnet model"""
    def __init__(self, win_size: Optional[int]=None, filter_ratio: Optional[float]=None, 
                 using_depth: Optional[bool]=None, using_sam: Optional[bool]=None,
                 finetune: Optional[bool]=None, binary_thresh: Optional[float]=None,
                 pretrain_sam: Optional[str]=None, head: Optional[object]=None):
        super().__init__()

        self.hitnet = Hitnet()
        self.batch = 0
        self.ssim = SSIM()

    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image).cuda()
        return image.permute(2, 0, 1).contiguous()  
    
    def _filter(self, input):
        batch_size, c, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, c, h * w), dim=2)
        return self.filter_ratio * threshold.contiguous().view(batch_size, c, 1, 1)
    
    def find_bbox(self, feat_map):
        feat_map[feat_map < self.binary_thresh] = 0
        feat_map = feat_map.squeeze().unsqueeze(-1).clone().detach().cpu().numpy()
        feat_map = (feat_map*255).astype('uint8')
        contours, hierarchy = cv2.findContours(feat_map,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]

        max_index = areas.index(max(areas))
        max_contour = contours[max_index]
        x,y,w,h = cv2.boundingRect(max_contour)
 
        return np.array([x, y, x + w, y + h])
    
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

    def cal_loss(self, preds: torch.Tensor, gts: torch.Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(gts, kernel_size=31, stride=1, padding=15) - gts)
        wbce = F.binary_cross_entropy_with_logits(preds, gts, reduction='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        preds = torch.sigmoid(preds)
        inter = ((preds * gts)*weit).sum(dim=(2, 3))
        union = ((preds + gts)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def compute_surface_normals(self, depth_map):
        # 计算深度图的梯度
        dzdx = torch.gradient(depth_map, dim=2)[0]
        dzdy = torch.gradient(depth_map, dim=3)[0]
        # 计算法向量的x、y、z分量
        normal_x = -dzdx
        normal_y = -dzdy
        normal_z = torch.ones_like(depth_map)
        # 归一化法向量
        norm = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= norm
        normal_y /= norm
        normal_z /= norm
        return torch.cat((normal_x, normal_y, normal_z), dim=1)

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True

    def forward(self, raw, input, label, depth, mode='loss'):  
        
        if isinstance(input, (tuple, list)): 
            input = torch.stack(input, dim=0)
        if isinstance(label, (tuple, list)):         
            label = torch.stack(label, dim=0)   
        depth = torch.stack(depth, dim=0)
        surface_normals = self.compute_surface_normals(depth)
        if mode == 'loss':             
            # for name, param in self.named_parameters():
            #     print(name)
            # import pdb;pdb.set_trace()



            # HitNet
            # P1, P2 = self.hitnet(input, tokens[-1])
            embedding1, P1, P2 = self.hitnet(input, depth)#depth)
            # output = F.upsample(P1[-1] + P2, size=label.shape[-2:], mode='bilinear', align_corners=False)
            losses = [self.cal_loss(preds=out, gts=label) for out in P1]
            loss_p1=0
            gamma=0.2
            for it in range(len(P1)):
                loss_p1 += (gamma * it) * losses[it]
            loss_P2 = self.cal_loss(preds=P2, gts=label)
            embedding1 = (embedding1 - embedding1.min()) / (embedding1.max() - embedding1.min() + 1e-8)
            loss_3 = self.ssim(embedding1, input) 
            loss = loss_p1 + loss_P2 + loss_3 * 0.1#0.01
            return {'loss': loss}#self.cal_loss(preds=output, gts=label)} 
 
                 
        elif mode == 'predict':


            embedding1, P1, P2 = self.hitnet(input, depth)#tokens[-1])
            output = F.interpolate(P1[-1] + P2, size=label.shape[-2:], mode='bilinear', align_corners=False)

            # # Ensure the directory for saving images exists, otherwise create it
            # from torchvision.utils import save_image
            # output_dir = 'visualizations_our_'
            # os.makedirs(output_dir, exist_ok=True)
            # # Function to save an image tensor
            # def save_tensor_as_image(tensor, name):
            #     file_path = os.path.join(output_dir, f'{name}.png')
            #     # If the tensor has 1 channel, convert it to 3 channels for saving as an image
            #     if tensor.size(1) == 1:
            #         tensor = tensor.repeat(1, 3, 1, 1)
            #     save_image(tensor, file_path)
            #     return file_path
            # def denormalize(tensor, mean, std):
            #     if tensor.ndim == 4:  # Batch of images of shape (B, C, H, W)
            #         mean = mean[None, :, None, None]
            #         std = std[None, :, None, None]
            #     elif tensor.ndim == 3:  # Single image of shape (C, H, W)
            #         mean = mean[:, None, None]
            #         std = std[:, None, None]
            #     tensor = tensor * std + mean
            #     tensor = tensor.clamp(0, 1)  # Ensure the values are in [0, 1]
            #     return tensor        
            # # Save images
            # mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
            # std = torch.tensor([0.229, 0.224, 0.225]).cuda()
            # input_denorm = denormalize(input, mean, std)
            # name = raw[0].split('/')[-1].split('.')[0]
            # input_path = save_tensor_as_image(input_denorm, f'{name}_input')
            # label_path = save_tensor_as_image(label, f'{name}_label')
            # output_path = save_tensor_as_image(output.sigmoid(), f'{name}_output')
            # depth_path = save_tensor_as_image(depth, f'{name}_depth')

            return output.sigmoid(), label
        elif mode == 'tensor':
            # Return tensors for visualization
            return output
        else:
            raise NotImplementedError(f'Unsupported mode {mode}')
        



    
@export
class our_2(Hook):
    """Init with pretrained model"""
    priority = 'NORMAL'

    def __init__(self):
        pass
    def before_train(self, runner):
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        # Load checkpoint for SOTA depth estimator
        # pretrain = 'pretrain/convlarge_hourglass_0.3_150_step750k_v1.1.pth'#'pretrain/hitnet.pth'#
        # checkpoint = torch.load(pretrain, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % pretrain)
        # if 'model_state_dict' in checkpoint:
        #     checkpoint = checkpoint['model_state_dict']
        # decoder_dict = {}
        # encoder_dict = {}
        # for key, value in checkpoint.items():
        #     if key.startswith('depth_model.decoder.'):
        #         new_key = key[20:]  
        #         decoder_dict[new_key] = value
        # msg = model.depth_head.load_state_dict(decoder_dict, strict=False)
        # print(msg)
        # for key, value in checkpoint.items():
        #     if key.startswith('depth_model.encoder.'):
        #         new_key = key[20:]  
        #         encoder_dict[new_key] = value
        # msg = model.depth_backbone.load_state_dict(encoder_dict, strict=False)
        # print(msg)


        # Load checkpoint of hitnet 
        pretrain = 'pretrain/pvt_v2_b2.pth'#'pretrain/hitnet.pth'#
        checkpoint = torch.load(pretrain, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrain)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        msg = model.hitnet.backbone.load_state_dict(checkpoint, strict=False)
        print(msg)

        # Load checkpoint of convnext 
        pretrain = 'pretrain/convnext_base_22k_224.pth'#'pretrain/hitnet.pth'#
        checkpoint = torch.load(pretrain, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrain)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        # msg = model.hitnet.backbone.prompt_encoder.encoder1.load_state_dict(checkpoint, strict=False)
        msg = model.hitnet.backbone.prompt_encoder.encoder2.load_state_dict(checkpoint, strict=False)
        # msg = model.hitnet.backbone.prompt_encoder.encoder2.load_state_dict(checkpoint, strict=False)
        print(msg)

        

        # # load pretrain for sam
        # sam_checkpoint = "pretrain/sam_vit_b_01ec64.pth"
        # checkpoint = torch.load(sam_checkpoint, map_location='cpu')
        # print("Load pre-trained checkpoint for sam from: %s" % sam_checkpoint)
        # msg = model.sam.load_state_dict(checkpoint, strict=False)
        # print(msg)

    # def before_val(self, runner):
    #     model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model

    #     # Load checkpoint of hitnet 
    #     pretrain = 'output/hardmard_nc4k/epoch_100.pth'
    #     checkpoint = torch.load(pretrain, map_location='cpu')
    #     print("Load pre-trained checkpoint from: %s" % pretrain)
    #     if 'model' in checkpoint:
    #         checkpoint = checkpoint['model']
    #     msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     print(msg)



        # checkpoint = torch.load(pretrain_sam, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % pretrain_sam)
        # if 'model' in checkpoint:
        #     checkpoint = checkpoint['model']
        # new_state_dict = {}
        # for key, value in checkpoint['state_dict'].items():
        #     if key.startswith('sam.'):
        #         new_key = key[4:]  
        #         new_state_dict[new_key] = value
        # msg = model.sam.load_state_dict(new_state_dict, strict=False)


class SSIM(torch.nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim(self, x, y):
        abs_diff = torch.abs(y - x)
        l1_loss = abs_diff.mean(1, True)
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean(1,True)
        l1_loss= 0.85 * ssim_loss + 0.15 * l1_loss
        
        return ssim_loss.mean()

    def forward(self, pred, target):
        return self._ssim(pred, target)



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#####
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#####------------------------------------
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class SAM(nn.Module):
    def __init__(self, ch_in=32, reduction=16):
        super(SAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc_wight = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )
        # self.normalize = normalize
        # self.num_s = int(plane_mid)
        # self.num_n = (mids) * (mids)
        # self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        #
        # self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        # self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        # self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x_h, x_l):
        #print('x_h shape, x_l shape,',x_h.shape, x_l.shape)
        b, c, _, _ = x_h.size()
        #print('self.avg_pool(x_h)',self.avg_pool(x_h).shape)
        y_h = self.avg_pool(x_h).view(b, c) # squeeze操作
        #print('***this is Y-h shape',y_h.shape)
        h_weight=self.fc_wight(y_h)
        #print('h_weight',h_weight.shape,h_weight) ##(batch,1)
        y_h = self.fc(y_h).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        #print('y_h.expand_as(x_h)',y_h.expand_as(x_h).shape)
        x_fusion_h=x_h * y_h.expand_as(x_h)
        x_fusion_h=torch.mul(x_fusion_h, h_weight.view(b, 1, 1, 1))
##################----------------------------------
        b, c, _, _ = x_l.size()
        y_l = self.avg_pool(x_l).view(b, c) # squeeze操作
        l_weight = self.fc_wight(y_l)
        #print('l_weight',l_weight.shape,l_weight)
        y_l = self.fc(y_l).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        #print('***this is y_l shape', y_l.shape)
        x_fusion_l=x_l * y_l.expand_as(x_l)
        x_fusion_l = torch.mul(x_fusion_l, l_weight.view(b, 1, 1, 1))
#################-------------------------------
        #print('x_fusion_h shape, x_fusion_l shape,h_weight shape',x_fusion_h.shape,x_fusion_l.shape,h_weight.shape)
        x_fusion=x_fusion_h+x_fusion_l

        return x_fusion # 注意力作用每一个通道上

#####
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs

        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x

class Hitnet(nn.Module):
    def __init__(self, channel=32,n_feat=32,scale_unetfeats=32,kernel_size=3,reduction=4,bias=False,act=nn.PReLU()):
        super(Hitnet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]

        # path = 'pretrain/pvt_v2_b2.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)



        # self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.decoder_level4 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level3 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level2 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level4 = nn.Sequential(*self.decoder_level4)

        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.decoder_level2 = nn.Sequential(*self.decoder_level2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.downsample_4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.decoder_level1 = [CAB(64, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)

        self.compress_out = BasicConv2d(2 * channel, channel, kernel_size=8, stride=4, padding=2)

        self.compress_out2 = BasicConv2d(2 * channel, channel, kernel_size=1)
        ##kernel_size, stride=1, padding=0, dilation=1



    def forward(self, x, pred_normal):

        # backbone
        embedding1, pvt = self.backbone(x, pred_normal)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

#############-------------------------------------------------
        # CIM
        # x1 = self.ca(x1) * x1 # channel attention
        # cim_feature = self.sa(x1) * x1 # spatial attention

        cim_feature = self.decoder_level1(x1)
        ####
        # CFM
        x2_t = self.Translayer2_1(x2)#####channel=32
        x3_t = self.Translayer3_1(x3) ####channel=32
        x4_t = self.Translayer4_1(x4)

####stage 1--------------------------------------------------
        stage_loss=list()
        cfm_feature=None
        for iter in range(4):
            # print('iter',iter)
            # print(x4_t.shape,cfm_feature)
            if cfm_feature==None:
                x4_t=x4_t
            else:
                # x4_t=torch.cat((x4_t, self.downsample_4(cfm_feature)), 1)
                x4_t = torch.cat((self.upsample_4(x4_t), cfm_feature), 1)
                x4_t = self.compress_out(x4_t)
                # x4_t = x4_t+self.downsample_4(cfm_feature)
                # x4_t=x4_t*self.downsample_4(cfm_feature)
                # print(self.downsample_4(cfm_feature).shape,x4_t.shape)
            # if cfm_feature!=0:
            #     x4_t
            x4_t_feed = self.decoder_level4(x4_t)  ######channel=32, width and height
            x3_t_feed = torch.cat((x3_t, self.upsample(x4_t_feed)), 1)
            x3_t_feed = self.decoder_level3(x3_t_feed)
            if iter>0:
                # print('here',iter)
                x2_t=torch.cat((x2_t, cfm_feature), 1)
                x2_t=self.compress_out2(x2_t)
            x2_t_feed = torch.cat((x2_t, self.upsample(x3_t_feed)), 1)
            x2_t_feed=self.decoder_level2(x2_t_feed) ####(3 channel, 3channel)
            cfm_feature = self.conv4(x2_t_feed)
            prediction1 = self.out_CFM(cfm_feature)
            # print('this is prediction shape',prediction1.shape)
            prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')
            stage_loss.append(prediction1_8)
###-----------------------
        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        # prediction1 = self.out_CFM(cfm_feature )
        prediction2 = self.out_SAM(sam_feature)
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        return embedding1, stage_loss, prediction2_8



import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x


    
# main module of Texture diffuser
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ShapePropWeightRegressor(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(ShapePropWeightRegressor, self).__init__()
        use_gn = False
        self.latent_dim = latent_dim
        self.reg = nn.Conv2d(in_channels, self.latent_dim*49, kernel_size=1)

    def forward(self, x):
        weights = self.reg(x)
        return torch.sigmoid(weights)





class EncoderBlock(nn.Module):
    def __init__(self, in_channels):
        super(EncoderBlock, self).__init__()
        self.encoder1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.encoder2 = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6),
            nn.Linear(in_channels, 4 * in_channels),
            nn.GELU(),
            nn.Linear(4 * in_channels, in_channels),
        )
    def forward(self, x):
        embedding = self.encoder1(x).permute(0, 2, 3, 1)
        embedding = self.encoder2(embedding).permute(0, 3, 1, 2)
        embedding = embedding + x
        return embedding        

class convnext_Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ShapePropEncoder(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(ShapePropEncoder, self).__init__()
        # self.adapter = nn.Conv2d(in_channels,3,1)
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # dims = [dim//8,dim//4,dim//2,dim]
        dims = [128, 256, 512, 1024]#[96, 192, 384, 768]
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks

        drop_path_rate=0.4
        depths = [3, 3, 27, 3]#[3, 3, 9, 3]
        # depths = [1,1,3,1]

        layer_scale_init_value=1.0
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[convnext_Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        ####################
        ##decoder_head#####
        ####################
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(nn.Conv2d(dims[i], out_dim, 1))
        self.fusion_conv = nn.Conv2d(out_dim*4, out_dim, 1)     

    def forward(self, x):
        # x = self.adapter(x)
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)

        tmp=[]
        for i in range(len(outs)):
            x = outs[i]
            conv = self.convs[i]
            tmp.append(F.interpolate(conv(x), size=outs[0].shape[2:], mode='bilinear'))
        final = self.fusion_conv(torch.cat(tmp, dim=1))   
        return final


class MessagePassing(nn.Module):
    def __init__(self, latent_dim, img_size=384, k=7, max_step=2, sym_norm=False):
        super(MessagePassing, self).__init__()
        self.k = k
        self.size = k * k
        self.max_step = max_step
        self.sym_norm = sym_norm
        self.img_size = img_size
        self.conv = nn.Conv2d(latent_dim, 3, 1)
    def forward(self, input, weight):
        eps = 1e-5
        n, c, h, w = input.size()
        wc = weight.shape[1] // self.size

        weight = weight.view(n, wc, self.size, h * w)
        if self.sym_norm:
            # symmetric normalization D^(-1/2)AD^(-1/2)
            D = torch.pow(torch.sum(weight, dim=2) + eps, -1/2).view(n, wc, h, w)
            D = F.unfold(D, kernel_size=self.k, padding=self.padding).view(n, wc, self.window, h * w) * D.view(n, wc, 1, h * w)
            norm_weight = D * weight
        else:
            # random walk normalization D^(-1)A
            norm_weight = weight / (torch.sum(weight, dim=2).unsqueeze(2) + eps)
        x = input
        for i in range(max(h, w) if self.max_step < 0 else self.max_step):
            x = F.unfold(x, kernel_size=self.k, padding=3).view(n, c, self.size, h * w)
            x = (x * norm_weight).sum(2).view(n, c, h, w)
        x =  self.conv(x)
        x = F.interpolate(x, size=(self.img_size,self.img_size), mode='bilinear')
        return x 

class ShapePropDecoder(nn.Module):
    def __init__(self, out_dim, latent_dim):
        super(ShapePropDecoder, self).__init__()
        use_gn = False
        # latent_dim = 24
        dilation = 1
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.ReLU(True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.ReLU(True),
            nn.Conv2d(latent_dim, out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            # nn.Conv2d(latent_dim, out_dim, kernel_size=3) # depthwise conv
        )
    def forward(self, embedding):
        x = self.decoder(embedding)
        return x

class prompt_encoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, depth, fusion=False):
        super(prompt_encoder, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # self.shared_conv = nn.Conv2d(self.embed_dim//self.scale_factor, self.embed_dim, kernel_size=3,padding=1)
        # self.depth_adapter = nn.Sequential(
        #     nn.Conv2d(1, self.embed_dim//self.scale_factor, kernel_size=3, padding=1)
        # )
        # for i in range(self.depth):
        #     lightweight_mlp = nn.Sequential(
        #         # nn.GELU(),
        #         nn.Conv2d(self.embed_dim//self.scale_factor,self.embed_dim//self.scale_factor, kernel_size=3, padding=1),#self.input_dim//self.scale_factor, self.input_dim//self.scale_factor),
        #         nn.ReLU(),
        #     )
        #     setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        # propagation model
        self.propagation_weight_regressor = ShapePropWeightRegressor(3, latent_dim)
        # self.encoder0 = ShapePropEncoder(4, latent_dim//3)
        self.encoder1 = nn.Conv2d(1, latent_dim, 1)
        self.encoder2 = ShapePropEncoder(3, 24)
        # self.adaptor = nn.Conv2d(6,3,1)
        self.message_passing = MessagePassing(latent_dim, img_size=384, sym_norm=False)

        self.freq_nums = 0.3
        
    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).cuda()
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask)
        fr = fft.real
        fi = fft.imag
        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)
        return inv   


    def init_embeddings(self, x):
        x = x.permute(0,3,1,2).contiguous()
        N, C, H, W = x.shape
        x = x.reshape(N, C, H*W).permute(0, 2, 1)
        return self.embedding_generator(x)


    def forward(self, image, cues, cross=False):

        H = 12
        x = F.interpolate(image, size=([H,H]), mode='bilinear')
        # cues = F.interpolate(cues, size=([H,H]), mode='bilinear')


        x = self.fft(x, self.freq_nums)

        # x = self.prompt_generator(x)#.flatten(2).permute(0, 2, 1)
        prompts = []


        # propagation  
        weights = self.propagation_weight_regressor(x)
        embedding1 = self.encoder1(cues)
        embedding2 = self.message_passing(F.interpolate(embedding1, size=(H,H), mode='bilinear'), weights)
        # embedding0 = self.encoder0(torch.cat([cues, image], dim=1))
 
        # embedding3 = self.encoder2(self.adaptor(torch.cat([embedding2, image], dim=1)))
        embedding3 = self.encoder2(embedding2+image)

        

        return embedding2, embedding3

class prompt_decoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, depth, fusion=False):
        super(prompt_decoder, self).__init__()
        self.depth = depth
        # for i in range(depth):
        self.decoder = nn.Sequential(*[ShapePropDecoder(embed_dim, 24) for i in range(depth)])
            # setattr(self, 'decoder_{}'.format(str(i)), decoder)

    def forward(self, embedding, cross=False):
        # adapter
        prompts=[]
        for i in range(self.depth):
            prompt = self.decoder[i](embedding)
            prompt = prompt#torch.cat((torch.zeros(B,1,C).cuda(), prompt.flatten(2).permute(0,2,1)), dim=1)
            prompts.append(prompt)
        return prompts 
# class prompt_decoder(nn.Module):
#     def __init__(self, latent_dim, embed_dim, depth, fusion=False):
#         super(prompt_decoder, self).__init__()
#         self.depth = depth
#         self.decoder = ShapePropDecoder(embed_dim, 24)

#     def forward(self, embedding, cross=False):
#         # adapter
#         prompts=[]
#         prompt = self.decoder(embedding)
#         for i in range(self.depth):
#             prompt = prompt
#             prompts.append(prompt)
#         return prompts 


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # for texture diffuser
        self.latent_dim = 24
        self.prompt_encoder = prompt_encoder(self.latent_dim, embed_dims, depths, True) 
        self.prompt_decoder = nn.Sequential(*[prompt_decoder(self.latent_dim, embed_dims[i], depths[i], True) for i in range(len(depths))])
        # nn.Sequential(*[prompt_decoder(self.scale_factor, embed_dims[i], depths[i], True) for i in range(len(depths))])

        self.apply(self._init_weights)
        self.batch = 0
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



    def forward_features(self, x, depth):
        self.batch += 1

        B = x.shape[0]
        outs = []

        # stage 1
        image=x
        x, H, W = self.patch_embed1(x)

        # depth = F.interpolate(pred_normal, size=(88, 88), mode='bilinear')
        B,N,C = x.shape
        embedding1, embedding3 = self.prompt_encoder(image, depth)

        prompt = self.prompt_decoder[0](embedding3)
        for i, blk in enumerate(self.block1):
            prompt[i] = F.interpolate(prompt[i], size=(H,W), mode='bilinear').flatten(2).permute(0,2,1).reshape(x.shape)
            x = blk(x+prompt[i], H, W)#+depth[i], H, W) #10, 176^2, 64
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)       

        prompt = self.prompt_decoder[1](embedding3)
        for i, blk in enumerate(self.block2):
            prompt[i] = F.interpolate(prompt[i], size=(H,W), mode='bilinear').flatten(2).permute(0,2,1).reshape(x.shape)
            x = blk(x+prompt[i], H, W)#+depth[i], H, W) #10, 176^2, 64
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)

        prompt = self.prompt_decoder[2](embedding3)
        for i, blk in enumerate(self.block3):
            prompt[i] = F.interpolate(prompt[i], size=(H,W), mode='bilinear').flatten(2).permute(0,2,1).reshape(x.shape)
            x = blk(x+prompt[i], H, W)#+depth[i], H, W) #10, 176^2, 64
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)

        prompt = self.prompt_decoder[3](embedding3)
        for i, blk in enumerate(self.block4):
            prompt[i] = F.interpolate(prompt[i], size=(H,W), mode='bilinear').flatten(2).permute(0,2,1).reshape(x.shape)
            x = blk(x+prompt[i], H, W)#+depth[i], H, W) #10, 176^2, 64
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return embedding1, outs

        # return x.mean(dim=1)

    def forward(self, x, depth):
        embedding1, x = self.forward_features(x, depth)
        # x = self.head(x)

        return embedding1, x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict






class new_WindowFusion(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        ix = x; iy = y
        H = int(math.sqrt(N))
        q, k = self.qk(x).reshape(B, N, 2, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(y).reshape(B, N, 1, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)[0]                                

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + ix + iy
        x = x.permute(0,2,1).reshape(B,C,H,H)
        return x

class WindowFusion(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size=(10, 10), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        q_size = window_size[0]
        kv_size = window_size[1]
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)    
        # self.att_project = nn.Linear(num_heads, 1)    
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x,y):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        H= int(math.sqrt(N))
        W= H
        x = x.reshape(B_, H, W, C)
        identity = x.permute(0,3,1,2)
        y = y.reshape(B_, H, W, C)
        identity_y = y.permute(0,3,1,2)
        x = x

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        y = window_partition(y, self.window_size[0])
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C
        y = y.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C
        B_w = x.shape[0]
        N_w = x.shape[1]
        kv = self.kv(y).reshape(B_w, N_w, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B_w, N_w, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 
        q = q[0] #nW*B, heads, window_size*window_size, C/heads(144,8,64,96)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C


        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)
        x = x.permute(0,2,1)
        x = x.view(B_,C,H,W)



        return x*identity_y+identity_y, x.sigmoid()#bias



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

@register_model
class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



@register_model
class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@register_model
class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)