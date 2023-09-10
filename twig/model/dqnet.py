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
class DQnet(BaseModel):
    """DQnet model"""
    def __init__(self, win_size: Optional[int]=None, filter_ratio: Optional[float]=None, 
                 using_depth: Optional[bool]=None, using_sam: Optional[bool]=None,
                 finetune: Optional[bool]=None, binary_thresh: Optional[float]=None,
                 pretrain_sam: Optional[str]=None, head: Optional[object]=None):
        super().__init__()
        # self.last_ch = 128
        # self.space_encoder = resnet50()
        # self.higher_encoder = vit_base_patch16_224()
        # self.segmentation_head =  FPNHEAD()

        # ViT_integral(img_size=384, patch_size=16, embed_dim=768,
        #                            depth=12, num_heads=12, drop_path_rate=0.1)
        # swin(img_size=384, patch_size=4, window_size=12, 
        #                     embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))

        # ViT_integral(img_size=384, patch_size=16, embed_dim=768,
        #                            depth=12, num_heads=12, drop_path_rate=0.1)
        # self.higher_encoder = vit_base_patch16_224()  
        # self.higher_encoder = integral_ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
        # ViT_integral(img_size=384, patch_size=16, embed_dim=768,
        #                    depth=12, num_heads=12, drop_path_rate=0.1) 
        # swin(img_size=384, patch_size=4, window_size=12, 
        #                             embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))


        

        # self.swin = swin()
        # self.segmentation_head =  head#Decoder(128)#SimpleHead(1)#FPNHEAD()#MultiScaleMaskedTransformerDecoder(in_channels=512,
            # num_classes=1, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048,
            # dec_layers=10, pre_norm=False, mask_dim=256, enforce_input_project=False)##vitHead(1)#
        #vitHead(num_classes=1)
        # Mask2FormerHead(num_classes=1, 
        # in_channels=[2048,1024,512,256], out_channels=256, feat_channels=256,
        # pixel_decoder=None, transformer_decoder=None)


        # using_depth
        # self.using_depth = using_depth

        # self.depth_estimator = dinov2_depth_estimator()
        # self.min_depth = 1e-3
        # self.max_depth = 10
        # self.n_bins=256
        # self.depth_head = NewCRFDepth()
        # self.norm0 = nn.LayerNorm(768)
        # self.norm1 = nn.LayerNorm(768)
        # self.norm2 = nn.LayerNorm(768)

        # self.depth_backbone = convnext_large(layer_scale_init_value=1.0, drop_path_rate=0.4, in_22k=True, out_indices=[0, 1, 2, 3])
        # self.depth_head = HourglassDecoder()


        self.hitnet = Hitnet()

        # using_sam
        # self.win_size = win_size
        # self.filter_ratio = filter_ratio
        # sam_checkpoint = 'pretrain/sam_vit_b_01ec64.pth'#"pretrain/sam_vit_h_4b8939.pth"
        # self.pretrain_sam = sam_checkpoint
        # model_type = "vit_b"
        # self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)          
        # self.resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        # counting
        self.batch = 0

    def encoder_translayer(self, raw, x):
        higher_feature = self.higher_encoder(x) # 16, 768, 24, 24
        space_feature = self.space_encoder(x, higher_feature, mode='double')
        # space_feature = self.space_encoder(x)#, mode='single')
        # space_feature = self.space_encoder(x, higher_feature)        
        return space_feature, higher_feature
    
    def body(self, raw, input):
        space_feature, higher_feature = self.encoder_translayer(raw, input)
        # higher_feature = self.encoder_translayer(raw, input)

        over_all_feature = space_feature[1:]#+[higher_feature]# higher_feature[-1:]
        logits = self.segmentation_head(over_all_feature)
        return logits
    
    def cal_loss(self, preds: torch.Tensor, gts: torch.Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(gts, kernel_size=31, stride=1, padding=15) - gts)
        wbce = F.binary_cross_entropy_with_logits(preds, gts, reduction='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        preds = torch.sigmoid(preds)
        inter = ((preds * gts)*weit).sum(dim=(2, 3))
        union = ((preds + gts)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()


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
    def forward(self, raw, input, label, depth, mode='loss'):  
        # using_sam = self.using_sam
        if isinstance(input, (tuple, list)): 
            # if self.finetune==True:     
            #     input = torch.stack(input, dim=0).squeeze(1)
            # elif self.finetune==False:
            #     input = torch.stack(input, dim=0)
            # sam_input = torch.stack(sam_input, dim=0).squeeze(1)
            input = torch.stack(input, dim=0)
        if isinstance(label, (tuple, list)):         
            # if self.finetune==True:
            #     label = np.stack(label, axis=0)
            # elif self.finetune==False:
            label = torch.stack(label, dim=0)   
        depth = torch.stack(depth, dim=0)
        surface_normals = self.compute_surface_normals(depth)
        if mode == 'loss':
            # Call the forward and return loss
            # if self.finetune==False:
            #     output = self.body(raw, input)
            #     return {'loss': self.cal_loss(preds=output, gts=label)} 

            
                # indices = torch.nonzero(label)
                # stride=4
                # filtered_indices = indices[(indices[:, 2] % stride == 0) & (indices[:, 3] % stride == 0)][:,[0,2,3]]
                # coords=[]
                # for i in range(max(filtered_indices[:, 0])+1):
                #     coords.append(filtered_indices[filtered_indices[:, 0]==i][:,1:]) # fetch coords in each sample for one batch
                # coords_label = torch.ones(input.shape[0], 1)   
              
            # finetune with depth and normal
            # self.sam.train()
            # sam_input = []
            # for i, image in enumerate(raw):
            #     sam_input.append(self.prepare_image(np.array(image), self.resize_transform, self.sam))
            # sam_input = torch.stack(sam_input, dim=0)
            # sam_input = self.sam.preprocess(sam_input)

                # training for depth head
                # tokens = self.depth_estimator(input)
                # B,C,H,W = tokens[0].shape
                # tokens[0] = self.norm0(tokens[0].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
                # tokens[1] = self.norm1(tokens[1].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
                # tokens[2] = self.norm2(tokens[2].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
                # pred_depth = self.fuse(torch.cat(tokens, dim=1))
                # pred_depth = pred_depth.reshape(B,C,H*W).permute(0,2,1)
                # pred_depth = self.depth_head(pred_depth).reshape(B,1,H,W)
                # return {'loss': nn.MSELoss()(pred_depth, depth)}
            # tokens = self.depth_estimator(input)
            # # tokens = self.bottleneck(tokens)
            # B,C,H,W = tokens[0].shape
            # tokens[0] = self.norm0(tokens[0].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
            # tokens[1] = self.norm1(tokens[1].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
            # tokens[2] = self.norm2(tokens[2].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W) 

            # with torch.no_grad():
            #     depth = self.depth_backbone(input)
            #     import pdb;pdb.set_trace()
            #     depth = self.depth_head(depth)['prediction']
            #     import pdb;pdb.set_trace()

            # pred_depth_1 = (self.depth_head(input))
            # pred_depth = self.compute_surface_normals(pred_depth_1)

            if self.batch % 10 == 0:
                image = depth[0].detach().cpu().numpy().squeeze()
                # image = (image-image.min())/(image.max()-image.min())
                # pil_image = Image.fromarray((image * 255).astype('uint8'), mode='L')
                # pil_image.save(f'visualize/normal/{self.batch}pred.jpg')
                plt.imsave(f'visualize/normal/{self.batch}pred.jpg', (10 - image) / 10, cmap='plasma')
            self.batch+=1

            # HitNet
            # P1, P2 = self.hitnet(input, tokens[-1])
            P1, P2 = self.hitnet(input, depth)#depth)
            # output = F.upsample(P1[-1] + P2, size=label.shape[-2:], mode='bilinear', align_corners=False)
            losses = [self.cal_loss(preds=out, gts=label) for out in P1]
            loss_p1=0
            gamma=0.2
            for it in range(len(P1)):
                loss_p1 += (gamma * it) * losses[it]
            loss_P2 = self.cal_loss(preds=P2, gts=label)
            loss = loss_p1 + loss_P2
            return {'loss': loss}#self.cal_loss(preds=output, gts=label)} 

            # image_embedding = self.sam.image_encoder(sam_input, depth)#surface_normals)
            # # coords = torch.stack([coord[torch.randint(low=0,high=coord.shape[0],size=(1,))] for coord in coords], dim=0) #(4,1,2)
            # sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            #     points=None,
            #     boxes=None,
            #     masks=None,
            # )
            # low_res_masks, iou_predictions = self.sam.mask_decoder(
            # image_embeddings=image_embedding,
            # image_pe=self.sam.prompt_encoder.get_dense_pe(),
            # sparse_prompt_embeddings=sparse_embeddings,
            # dense_prompt_embeddings=dense_embeddings,
            # multimask_output=False,
            # )
            # upscaled_masks = self.sam.postprocess_masks(low_res_masks, (1024,1024), (384,384)).cuda()
            # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
            # gt_mask_resized = label
            # gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            # return {'loss': self.cal_loss(preds=upscaled_masks, gts=gt_binary_mask)}  
                 
        elif mode == 'predict':
            # Return tensors used in the metric computation
            # if using_sam == False:
            #     output = self.body(raw, input)
                # self.train()
                # cam_extractor = CAM(self, target_layer='segmentation_head.cls') 

            # traverse the point in mask
            indices = torch.nonzero(label)
            stride=4
            filtered_indices = indices[(indices[:, 2] % stride == 0) & (indices[:, 3] % stride == 0)][:,[0,2,3]]
            coords=[]
            for i in range(max(filtered_indices[:, 0])+1):
                coords.append(filtered_indices[filtered_indices[:, 0]==i][:,1:]) # fetch coords in each sample for one batch
            coords_label = torch.ones(1)
                # coords is a list, each element of shape [sample_num, 2]


                # output = torch.sigmoid(self.body(raw, input))
                
                # with torch.set_grad_enabled(True):
                #     output = self.body(raw, input)
                #     bbox_coords=[]

                #     # using dqnet output as bbox prompt
                #     try:
                #         for out in output:
                #             bbox_coords.append(self.find_bbox(out.sigmoid()))
                #         bbox_coords = torch.from_numpy(np.stack(bbox_coords, axis=0)).cuda()
                #     except:
                #         pass
                    # using label as bbox prompt
                    # for l in label:
                    #     l = l.permute(1,2,0).clone().detach().cpu().numpy().astype('uint8')
                    #     contours, hierarchy = cv2.findContours(l,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
                    #     areas = [cv2.contourArea(c) for c in contours]
                    #     max_index = areas.index(max(areas))
                    #     max_contour = contours[max_index]
                    #     x,y,w,h = cv2.boundingRect(max_contour)
                    #     bbox_coords.append(np.array([x, y, x + w, y + h])) 
                    # bbox_coords = torch.from_numpy(np.stack(bbox_coords, axis=0)).cuda()

                    # for i, out in enumerate(output):
                    #     out = out.squeeze().unsqueeze(-1).clone().detach().cpu().numpy()>0
                    #     out = (out*255).astype('uint8')
                    #     import pdb; pdb.set_trace()
                    #     contours, hierarchy = cv2.findContours(np.array(out),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
                    #     areas = [cv2.contourArea(c) for c in contours]

                    #     max_index = areas.index(max(areas))
                    #     max_contour = contours[max_index]
                    #     x,y,w,h = cv2.boundingRect(max_contour)
                    #     bbox_coords.append(np.array([x, y, x + w, y + h]))

                    

                # topk
                # prompt_point = []
                # for i, x in enumerate(output):
                #     cam = cam_extractor(1, x)[0]# 第1类
                #     scale = output.shape[-1] / cam[i].shape[-1]
                    
                #     values, indices = torch.topk(cam[i].flatten(), 1)
                #     indices_y = (indices // cam[i].shape[-1] * scale).int()
                #     indices_x = (indices % cam[i].shape[-1] * scale).int()
                #     prompt_point.append(torch.stack([indices_x, indices_y], dim=1))
                #     prompt_label = torch.ones(len(indices_y))


                    # Visualize the CAM and the max point
                    # plt.imshow(cam[i].cpu().numpy()); 
                    # for j in range(len(indices_y)):
                    #     plt.scatter(indices_x[j].cpu()//scale, indices_y[j].cpu()//scale, s=50, c='r', marker='o') 
                    # plt.axis('off'); plt.tight_layout(); plt.savefig(f'{i}.png'); plt.close()

                # # local peak
                # prompt_point = []
                # prompt_label = []
                # cam = 1 - cam_extractor(0, output[0])[0].unsqueeze(1) # output[i] is all the same
                # scale = output.shape[-1] / cam[0].shape[-1]

                # assert self.win_size % 2 == 1, 'Window size for peak finding must be odd.'
                # offset = (self.win_size - 1) // 2
                # padding = torch.nn.ConstantPad2d(offset, float('-inf'))
                # padded_maps = padding(cam)
                # batch_size, c, h, w = padded_maps.size()
                # element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
                # element_map = element_map.to(cam.device)
                # _, indices  = F.max_pool2d(
                #     padded_maps,
                #     kernel_size = self.win_size, 
                #     stride = 1, 
                #     return_indices = True)
                # peak_map = (indices == element_map)  
                # mask = cam >= self._filter(cam)
                # peak_map = (peak_map & mask)
                # peak_list = torch.nonzero(peak_map)
                # for i in range(output.shape[0]):
                #     prompt_point.append((peak_list[peak_list[:,0]==i][:,2:] * scale)[:,[1,0]].int()) # flip the x, y coords
                #     prompt_label.append(torch.ones(prompt_point[i].shape[0]))

                # # using depth as mask prompt

                # depth = self.depth_estimator(input)
                # depth = torch.mean(depth, dim=1, keepdim=True)
                # depth = F.interpolate(self.resize_transform.apply_image_torch(depth), scale_factor=1/4)


                # output = self.body(raw, input)
                # prompt_box = []

                # try:
                #     for out in output:
                #         prompt_box.append(self.find_bbox(out.sigmoid()))
                #     # input = self.sam.preprocess(input)
                #     # image_embedding = self.sam.image_encoder(input)
                #     prompt_box = np.stack(prompt_box, axis=0)
                #     box = self.resize_transform.apply_boxes(prompt_box, (384,384))
                #     box_torch = torch.as_tensor(box, dtype=torch.float).cuda()

                #     # sam
                #     batched_input=[]
                #     for i, image in enumerate(raw):
                #         image = np.array(image)
                #         batched_input.append({
                #                 'image': self.prepare_image(image, self.resize_transform, self.sam),
                #                 # 'point_coords': coords[i].unsqueeze(0),#self.resize_transform.apply_coords_torch(prompt_point[i].cuda().unsqueeze(0), image.shape[:-1]),
                #                 # 'point_labels': coords_label[i].unsqueeze(0),#prompt_label[i].cuda().unsqueeze(0),
                #                 'boxes': box_torch[i].unsqueeze(0),#'self.resize_transform.apply_boxes_torch(bbox_coords[i], image.shape[:-1]),
                #                 'masks': depth,
                #                 'original_size': image.shape[:-1]
                #             })                       
                # except:

            # tokens = self.depth_estimator(input)
            # # tokens = self.bottleneck(tokens)
            # B,C,H,W = tokens[0].shape
            # tokens[0] = self.norm0(tokens[0].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
            # tokens[1] = self.norm1(tokens[1].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
            # tokens[2] = self.norm2(tokens[2].reshape(B,C,H*W).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)  
            # pred_depth_1 = (self.depth_head(input))
            # pred_depth = self.compute_surface_normals(pred_depth_1)

            P1, P2 = self.hitnet(input, depth)#tokens[-1])
            output = F.interpolate(P1[-1] + P2, size=label.shape[-2:], mode='bilinear', align_corners=False)

            # if self.batch % 20 == 0:
            #     image = pred_depth_1[0].permute(1, 2, 0).detach().cpu().numpy().squeeze()
            #     pil_image = Image.fromarray((image * 255).astype('uint8'), mode='L')
            #     pil_image.save(f'visualize/normal/{self.batch}pred.jpg')
            # self.batch+=1

            # for j, image in enumerate(raw):
            #     image = np.array(image)
            #     sam_input = self.prepare_image(image, self.resize_transform, self.sam)
            #     image_embedding = self.sam.image_encoder(self.sam.preprocess(sam_input).unsqueeze(0), tokens)#surface_normals)
            #     #     iou = []
            #     #     # for i in tqdm(range(coords[j].shape[0])):
            #     #     #     # batched_input=[]
            #     #     #     # batched_input.append({
            #     #     #     #         'image': self.prepare_image(image, self.resize_transform, self.sam),
            #     #     #     #         'point_coords': self.resize_transform.apply_coords_torch(coords[j][i].cuda().unsqueeze(0).unsqueeze(0), image.shape[:-1]),
            #     #     #     #         'point_labels': coords_label.unsqueeze(0),#prompt_label[i].cuda().unsqueeze(0),
            #     #     #     #         # 'boxes': box_torch[i].unsqueeze(0),#'self.resize_transform.apply_boxes_torch(bbox_coords[i], image.shape[:-1]),
            #     #     #     #         # 'masks': depth,
            #     #     #     #         'original_size': image.shape[:-1]
            #     #     #     #     })   
                


            #     #     #     # batched_output = self.sam(batched_input, multimask_output=False)
            #     #     #     sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            #     #     #         points=[self.resize_transform.apply_coords_torch(coords[j][i].cuda().unsqueeze(0).unsqueeze(0), image.shape[:-1]),
            #     #     #                 coords_label.unsqueeze(0)],
            #     #     #         boxes=None,
            #     #     #         masks=None,
            #     #     #     )
            #     #     #     low_res_masks, iou_predictions = self.sam.mask_decoder(
            #     #     #     image_embeddings=image_embedding,
            #     #     #     image_pe=self.sam.prompt_encoder.get_dense_pe(),
            #     #     #     sparse_prompt_embeddings=sparse_embeddings,
            #     #     #     dense_prompt_embeddings=dense_embeddings,
            #     #     #     multimask_output=False,
            #     #     #     )
            #     #     #     upscaled_masks = self.sam.postprocess_masks(low_res_masks, (1024,1024), (384,384)).cuda()
            #     #     #     batched_output = normalize(threshold(upscaled_masks, 0.0, 0))
            #     #     #     sam_output = batched_output[0].int()
            #     #     #     # sam_output = [x['masks'].int() for x in batched_output]
            #     #     #     # sam_output = torch.cat(sam_output, dim=0)
            #     #     #     intersection = (sam_output & (label[j]).int()).sum()
            #     #     #     union = (sam_output | (label[j]).int()).sum()
            #     #     #     iou.append(intersection.float() / union.float())
                    
            #     #     # matrix = np.zeros((384//stride, 384//stride))
            #     #     # num=0
            #     #     # for p, q in coords[j]:
            #     #     #     matrix[p//stride, q//stride] = iou[num]
            #     #     #     num+=1
            #     #     # # iou = [each.cpu() for each in iou]
            #     #     # # xnew = np.arange(0,384/stride); ynew = np.arange(0,384/stride)


            #     #     # # f = interp2d(coords[j][:,0].cpu().numpy(), coords[j][:,1].cpu().numpy(), iou, kind='cubic')
            #     #     # # xnew = np.arange(0,384); ynew = np.arange(0,384)
            #     #     # # znew = f(xnew, ynew)
            #     #     # # znew[label.cpu().squeeze()==0]=0
            #     #     # matrix = F.interpolate(torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0), size=(384,384), mode='bilinear')[0,0].numpy()

            #     #     # matrix_tensor = torch.tensor(matrix)

            #     #     # # if bbox_coords == []:
            #     #     # x=torch.tensor(0);y=torch.tensor(0);w = torch.tensor(matrix_tensor.shape[1])
            #     #     # heatmap_in_bbox = matrix_tensor
            #     #     # # else:
            #     #     # #     x=bbox_coords[0,0]; y=bbox_coords[0,1]; w=bbox_coords[0,-2]-x; h=bbox_coords[0,-1]-y
            #     #     # #     heatmap_in_bbox = matrix_tensor[y : y+h, x : x+w]
            #     #     # max_position = torch.argmax(heatmap_in_bbox.reshape(1, -1), 1)
            #     #     # max_y = max_position / w.cpu()
            #     #     # max_x = max_position % w.cpu()
            #     #     # max_y += y.cpu()
            #     #     # max_x += x.cpu()

            #     #     # min_position = torch.argmin(heatmap_in_bbox.reshape(1, -1), 1)
            #     #     # min_y = min_position / w.cpu()
            #     #     # min_x = min_position % w.cpu()
            #     #     # min_y += y.cpu()
            #     #     # min_x += x.cpu()

            # sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            #     points=None,#[self.resize_transform.apply_coords_torch(torch.zeros_like(prompt_point[0]).cuda().unsqueeze(0), image.shape[:-1]),
            #                 #prompt_label.unsqueeze(0)],
            #             # [self.resize_transform.apply_coords_torch(torch.stack((min_y,min_x),dim=1).cuda().unsqueeze(0), image.shape[:-1]),
            #             #  coords_label.unsqueeze(0)],
                        
                
            #     boxes=None,
            #     masks=None,
            # )
            # low_res_masks, iou_predictions = self.sam.mask_decoder(
            # image_embeddings=image_embedding,
            # image_pe=self.sam.prompt_encoder.get_dense_pe(),
            # sparse_prompt_embeddings=sparse_embeddings,
            # dense_prompt_embeddings=dense_embeddings,
            # multimask_output=False,
            # )
            # upscaled_masks = self.sam.postprocess_masks(low_res_masks, (1024,1024), (384,384)).cuda()
            # batched_output = normalize(threshold(upscaled_masks, 0.0, 0))
            # sam_output = batched_output[0].int()                    

            #     #     # visualization
            #     #     # plt.imshow(matrix, cmap='coolwarm', interpolation='bilinear')
            #     #     # # plt.savefig(f'iou_heatmap_{self.batch}.png')
            #     #     # plt.close()
            #     #     # # arr_norm = (matrix - matrix.min()) / (matrix.max() - matrix.min())
            #     #     # heatmap_pil = Image.fromarray(np.uint8(plt.cm.jet(matrix) * 255))
            #     #     # alpha = 0.5  # 混合比例
            #     #     # # import pdb; pdb.set_trace()
            #     #     # result = Image.blend(raw[j].convert('RGBA'), heatmap_pil, alpha)
            #     #     # new_image = Image.new("RGB", (raw[j].width * 4, raw[j].height))
            #     #     # transform = transforms.ToPILImage()
            #     #     # left = Image.blend(raw[j], transform(label.squeeze(0)).convert('RGB'), 0.3)
            #     #     # new_image.paste(raw[j], (0, 0))
            #     #     # new_image.paste(left, (raw[j].width, 0))
            #     #     # # dqnet_output = Image.fromarray(np.uint8(plt.cm.jet(output[0,0].cpu().numpy()) * 255))
            #     #     # new_image.paste(transform(torch.sigmoid(output).squeeze(0)), (2*raw[j].width, 0))
            #     #     # new_image.paste(heatmap_pil, (3*raw[j].width, 0))
            #     #     # new_image.save(f'visualize/4row_notune_nc4k/nc4k_4row_{self.batch}.png')

 



            #     #     # plt.imshow(image)
            #     #     # x = coords[j][:, 1].cpu(); y = coords[j][:, 0].cpu(); iou = [each.cpu() for each in iou]
            #     #     # plt.scatter(x, y, s=12, c=iou, cmap='coolwarm', vmin=0, vmax=1)
            #     #     # plt.axis('off')
            #     #     # plt.title('point importance')
            #     #     # plt.savefig(f'iou_{self.batch}.png')
            #     #     # plt.close()
            #     #         # for k, item in enumerate(sam_output):
            #     #         #     mask = item.repeat(3,1,1).permute(1,2,0).cpu().numpy()*255
            #     #         #     image = np.array(raw[k])
            #     #         #     alpha = 0.5
            #     #         #     blend = np.array(image) * alpha + mask * (1 - alpha)
            #     #         #     blend = np.clip(blend, 0, 255).astype('uint8') 
            #     #         #     # Visualize the mask on raw image
            #     #         #     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5.5))
            #     #         #     ax1.axis('off')
            #     #         #     ax2.axis('off')
            #     #         #     ax2.imshow(blend)
            #     #         #     ax2.set_title('sam predict')
            #     #         #     # Visualize the label
            #     #         #     gt = label[k].repeat(3,1,1).permute(1,2,0).cpu().numpy()*255
            #     #         #     gt_blend = np.array(image) * alpha + gt * (1 - alpha)
            #     #         #     gt_blend = np.clip(gt_blend, 0, 255).astype('uint8') 
            #     #         #     ax1.imshow(gt_blend)
            #     #         #     ax1.set_title('label')
            #     #         # # # Visualize the CAM and the max point
            #     #         # # ax2.imshow(cam[i][0].cpu().numpy()); 
            #     #         # # ax2.scatter(prompt_point[i][:, 0].cpu()//scale, prompt_point[i][:, 1].cpu()//scale, s=50, c='r', marker='o') 
            #     #         # # ax2.set_title('heatmap')
            #     #         # # plt.tight_layout(); plt.savefig(f'{i}_mask.png'); plt.close()
            #     #         # # # Visualize the bbox on raw image
            #     #         # # ax2.imshow(image)
            #     #         # # self.show_box(bbox_coords[i].cpu().numpy(), ax2)
            #     #         # # out = output[i].squeeze().clone().detach().cpu().numpy()>0
            #     #         # # self.show_mask(out, ax2)
            #     #         # # ax2.set_title('generated prompt')
            #     #         # plt.tight_layout(); plt.savefig(f'visualize/sam_after_bbox_finetune/{k+sam_output.shape[0]*self.batch}_mask.png'); plt.close()
            
            # return sam_output.unsqueeze(0), label 

            # visualize the predict without depth
            # for i in range(output.shape[0]):
            #     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5.5))
            #     ax1.axis('off')
            #     ax2.axis('off')
            #     image = np.array(raw[i])
            #     alpha = 0.5
            #     mask = output[i].sigmoid().repeat(3,1,1).permute(1,2,0).cpu().numpy()>0.5
            #     mask = 255 * mask
            #     blend = np.array(image) * alpha + mask * (1 - alpha)
            #     blend = np.clip(blend, 0, 255).astype('uint8')
            #     ax1.imshow(label[i].permute(1,2,0).cpu()); 
            #     ax1.set_title('label')
            #     ax2.imshow(blend)
            #     ax2.set_title('with')
            #     plt.tight_layout(); plt.savefig(f'visualize/with/with_{self.batch}.png'); plt.close()
            # self.batch+=1
            return output.sigmoid(), label
        elif mode == 'tensor':
            # Return tensors for visualization
            return output
        else:
            raise NotImplementedError(f'Unsupported mode {mode}')
        
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class integral_ResNet(timm.models.resnet.ResNet):
    def forward(self, x):
        x = self.forward_features(x)
        return x

# ResNet
class ResNet(timm.models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.cross0 = WindowFusion(768) 
        self.cross1 = WindowFusion(768)
        self.cross2 = WindowFusion(768)
        self.cross3 = WindowFusion(768)  

        self.convert0 = nn.Sequential(
            # nn.Conv2d(768, 64, 1)
            nn.ConvTranspose2d(768, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        )
        self.convert1 = nn.Sequential(
            # nn.Conv2d(768, 256, 1)
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.convert2 = nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2)#nn.Conv2d(768, 512, 1)#
        self.convert3 = nn.Conv2d(768, 1024, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        del self.fc      
        self.proj0 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4), nn.Conv2d(64, 768, kernel_size=1, stride=1, padding=0))
        self.proj1 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4), nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0))
        self.proj2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0))
        self.proj3 = nn.Sequential(nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0))     
        self.norm0 = nn.LayerNorm(768)
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.pos_norm0 = nn.BatchNorm2d(64)
        self.pos_norm1 = nn.BatchNorm2d(256)
        self.pos_norm2 = nn.BatchNorm2d(512)
        self.pos_norm3 = nn.BatchNorm2d(1024)

    def forward_features(self, x, y):
        features = []
        atts = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        features.append(x)
        x = self.maxpool(x) #c=64

        # # patch embed -> cross attention
        ys = [y.flatten(2).transpose(1,2) for i in range(4)]#[yy.flatten(2).transpose(1,2) for yy in y] # 16, 576, 768
        # ys =  self.c_verter(y)
        # ys = F.interpolate(ys, scale_factor=2)
        # ys = ys.flatten(2).transpose(1,2)
        xx = self.proj0(x)
        xx = xx.flatten(2).transpose(1, 2)
        xx = self.norm0(xx) # 20, 576, 768
        b,hh,c = xx.shape
        h = int(math.sqrt(hh))

        # cr, att = self.cross0(xx, ys[3])#ys[3], xx)#
        # atts.append(att)
        feat0 = self.relu(self.pos_norm0(self.convert0(y)))#cr)))
        x = self.layer1(feat0)
        features.append(x)

        # xx = self.proj1(x)
        # xx = xx.flatten(2).transpose(1, 2)
        # xx = self.norm1(xx)
        # cr, att = self.cross1(xx, ys[3])#
        # atts.append(att)
        feat1 = self.relu(self.pos_norm1(self.convert1(y)))#cr)))
        x = self.layer2(feat1)
        features.append(x)

        # xx = self.proj2(x)
        # xx = xx.flatten(2).transpose(1, 2)
        # xx = self.norm2(xx)      
        # cr, att = self.cross2(xx, ys[3])#ys[3], xx)#
        # atts.append(att)
        feat2 = self.relu(self.pos_norm2(self.convert2(y)))#cr)))
        x = self.layer3(feat2)
        features.append(x)

        # xx = self.proj3(x)
        # xx = xx.flatten(2).transpose(1, 2)
        # xx = self.norm3(xx)      
        # cr, att = self.cross3(xx, ys[3])#xx, ys[3])#ys[3], xx)#
        # atts.append(att)
        feat3 = self.relu(self.pos_norm3(self.convert3(y)))#cr)))
        x = self.layer4(feat3)
        features.append(x)

        return features
    
    def forward_plus(self, x, y):
        features = []
        atts = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        features.append(x)
        x = self.maxpool(x) #c=64

        # # patch embed -> cross attention
        # ys = [y.flatten(2).transpose(1,2) for i in range(4)]#[yy.flatten(2).transpose(1,2) for yy in y] # 16, 576, 768
        # ys =  self.c_verter(y)
        # ys = F.interpolate(ys, scale_factor=2)
        # ys = ys.flatten(2).transpose(1,2)
        xx = self.proj0(x)
        xx = xx.flatten(2).transpose(1, 2)
        xx = self.norm0(xx) # 20, 576, 768
        b,hh,c = xx.shape
        h = int(math.sqrt(hh))

        # cr, att = self.cross0(xx, ys[3])#ys[3], xx)#
        # atts.append(att)
        feat0 = self.relu(self.pos_norm0(self.convert0(y[0])))#cr)))
        x = self.layer1(feat0)
        features.append(x)

        # xx = self.proj1(x)
        # xx = xx.flatten(2).transpose(1, 2)
        # xx = self.norm1(xx)
        # cr, att = self.cross1(xx, ys[3])#
        # atts.append(att)
        feat1 = self.relu(self.pos_norm1(self.convert1(y[0])))#cr)))
        x = self.layer2(feat1)
        features.append(x)

        # xx = self.proj2(x)
        # xx = xx.flatten(2).transpose(1, 2)
        # xx = self.norm2(xx)      
        # cr, att = self.cross2(xx, ys[3])#ys[3], xx)#
        # atts.append(att)
        feat2 = self.relu(self.pos_norm2(self.convert2(y[1])))#cr)))
        x = self.layer3(feat2)
        features.append(x)

        # xx = self.proj3(x)
        # xx = xx.flatten(2).transpose(1, 2)
        # xx = self.norm3(xx)      
        # cr, att = self.cross3(xx, ys[3])#xx, ys[3])#ys[3], xx)#
        # atts.append(att)
        feat3 = self.relu(self.pos_norm3(self.convert3(y[2])))#cr)))
        x = self.layer4(feat3)
        features.append(x)

        return features
        
    def forward(self, x, y=None, z=None, mode=None):
        if mode=='double':
            features = self.forward_features(x, y)#self.forward_plus(x, y)#
        elif mode=='single':
            features = self.plain_forward(x)
        elif mode=='triple':
            features = self.forward_depth(x, y, z)
        return features

def resnet50(**kwargs):
    model = ResNet(
        block=Bottleneck, layers=[3, 4, 6, 3],  
        **kwargs
    )
    pretrain = 'pretrain/resnet50_pretrain.pth' 
    checkpoint = torch.load(pretrain, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % pretrain)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']

    # load pre-trained model
    import copy
    model_old = copy.deepcopy(model)
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model



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

    def __init__(self, dim, window_size=(24, 24), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        q_size = window_size[0]
        # kv_size = window_size[1]
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)    
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
        q = self.q(y).reshape(B_w, N_w, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]  # make torchscript happy (cannot use tensor as tuple)
        kv = self.kv(y).reshape(B_w, N_w, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 
        k, v = kv[0], kv[1] #nW*B, heads, window_size*window_size, C/heads(144,8,64,96)

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



        return x*(identity_y)+identity_y, x.sigmoid()



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

# DINOv2 depth estimation
class dinov2_depth_estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pretrain/dinov2', 'dinov2_vitb14', source='local') # sota depth estimation
        # self.linear_head = nn.Linear(2048, 1000)
    def forward(self, inputs):
        with torch.no_grad():
            inputs = F.interpolate(inputs, size=(392, 392), mode='bilinear', align_corners=False)#392, 392
            # output = self.model(inputs, is_training=True)['x_norm_patchtokens'].permute(0,2,1)
            # output = output.reshape(output.shape[0], output.shape[1], 28, 28)
            # output = F.interpolate(output, size=(384, 384), mode='bilinear', align_corners=False) 
            outputs = self.model(inputs, is_training=True)#['x_norm_patchtokens']
            outputs = [output['x_norm_patchtokens'].permute(0,2,1) for output in outputs]
            outputs = [output.reshape(output.shape[0], output.shape[1], 28, 28) for output in outputs]#28, 28
            # outputs = [F.interpolate(output, size=(384, 384), mode='bilinear', align_corners=False) for output in outputs]
        return outputs
        # return output

# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config

# Zoe depth estimation
class Zoe_estimator(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)
        # Zoe_N
        # conf = get_config("zoedepth", "infer")
        # self.model = build_model(conf)
        # self.model.eval()
        self.model = torch.hub.load("ZoeDepth-main", "ZoeD_N", pretrained=True, source="local")#torch.hub.load(repo, "ZoeD_N", pretrained=True)
        self.model.eval()
        self.batch = 0
    def forward(self, images):   
        # prepare image for the model

        # from torchvision.transforms import ToTensor
        predictions = []
        input = []
        for x in images:
            input.append(ToTensor()(x).unsqueeze(0).cuda())
        input = torch.cat(input,dim=0)
        with torch.no_grad():
            predicted_depth = self.model.infer(input)
        predictions.append(predicted_depth)
        import pdb;pdb.set_trace()
        predictions = torch.stack(predictions, dim=0).unsqueeze(1)

        # with torch.no_grad():
        #     predictions = self.model.infer(images)
        # import pdb;pdb.set_trace()
        
        # cal normal
        normals = []
        not_clip = []
        for i, pred in enumerate(predictions):
            import pdb;pdb.set_trace()
            output = np.array(pred[0].cpu())
            formatted = (output * 255 / np.max(output)).astype("uint8")
            image = np.array(Image.fromarray(formatted))
            image_depth = image.copy().astype(float)
            image_depth -= np.min(image_depth)
            image_depth /= np.max(image_depth)
            bg_threhold = 0.1
            x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            x[image_depth < bg_threhold] = 0
            y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            y[image_depth < bg_threhold] = 0
            z = np.ones_like(x) * np.pi * 2.0
            image = np.stack([x, y, z], axis=2)

            not_clip.append(torch.from_numpy(image).permute(2,0,1))

            image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
            image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            normals.append(torch.from_numpy(image).permute(2,0,1))

            # visualize the normal
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5.5))
            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(image)#s[i])
            ax1.set_title('image')
            ax2.imshow(image); 
            ax2.set_title('normal')
            plt.tight_layout(); plt.savefig(f'visualize/normal/normal_{i+predictions.shape[0]*self.batch}.png'); plt.close()
        self.batch+=1

        # return predictions
        # normals = torch.stack(normals, dim=0).cuda().type(torch.cuda.FloatTensor)
        not_clip = torch.stack(not_clip, dim=0).cuda().type(torch.cuda.FloatTensor)
        return not_clip#normals

# DPT depth estimation
class depth_estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("pretrain/DPT")#Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("pretrain/DPT")#Intel/dpt-large")
        self.batch = 0
        # self.model = torch.hub.load('checkpoints/facebookresearch', 'dinov2_vitl14', source='local') # sota depth estimation
    def forward(self, images):   
        # prepare image for the model
        predictions = []
        for x in images:
            inputs = self.image_processor(images=x, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].cuda()
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            predictions.append(predicted_depth)
        predictions = torch.stack(predictions, dim=0)
        # import pdb; pdb.set_trace()
        # cal normal
        normals = []
        for i, pred in enumerate(predictions):
            output = np.array(pred[0].cpu())
            formatted = (output * 255 / np.max(output)).astype("uint8")
            image = np.array(Image.fromarray(formatted))
            image_depth = image.copy().astype(float)
            image_depth -= np.min(image_depth)
            image_depth /= np.max(image_depth)
            bg_threhold = 0.1
            x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            x[image_depth < bg_threhold] = 0
            y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            y[image_depth < bg_threhold] = 0
            z = np.ones_like(x) * np.pi * 2.0
            import pdb;pdb.set_trace()
            image = np.stack([x, y, z], axis=2)
            image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
            image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            normals.append(torch.from_numpy(image).permute(2,0,1))
        # import pdb; pdb.set_trace()
            # # visualize the normal
            # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5.5))
            # ax1.axis('off')
            # ax2.axis('off')
            # ax1.imshow(images[i])
            # ax1.set_title('image')
            # ax2.imshow(image); 
            # ax2.set_title('normal')
            # plt.tight_layout(); plt.savefig(f'visualize/normal/normal_{i+predictions.shape[0]*self.batch}.png'); plt.close()
        self.batch+=1

        # return predictions
        normals = torch.stack(normals, dim=0).cuda().type(torch.cuda.FloatTensor)
        return normals


# ViT
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]#[:,1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
        checkpoint_model['pos_embed'] = checkpoint_model['pos_embed'][:,1:]   

class PatchEmbed(timm.models.layers.patch_embed.PatchEmbed):
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, depth=None, **kwargs):
        super().__init__(embed_layer=PatchEmbed, **kwargs)
        del self.head
        self.depth=depth
        self.fuse = nn.Conv2d(768*4, 768, kernel_size=1)
        self._norm0 = nn.LayerNorm(768)
        self._norm1 = nn.LayerNorm(768)
        self._norm2 = nn.LayerNorm(768)
    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.pos_embed
        cls_tokens = pos_embed[:, :1, :]
        pos_tokens = pos_embed[:, 1:, :]
        pos_tokens = pos_tokens.reshape(1, *self.patch_embed.grid_size, -1).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens,
            size=(H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]),
            mode='bicubic', align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat((cls_tokens, pos_tokens), dim=1)

        x = self.pos_drop(x + pos_embed)

        features=[]

        for i in range(self.depth):
            x = self.blocks[i](x)
            if i in [8,9,10,11]: #[2,5,8,11]:           
                features.append(x)
        # features[-1] = self.norm(features[-1])   
        return features


    def forward(self, x, strides=[4, 8, 16, 32]):

        features = self.forward_features(x)
        features[0] = self._norm0(features[0])
        features[1] = self._norm1(features[1])
        features[2] = self._norm2(features[2])
        features[3] = self.norm(features[3])
        features = [x[:, 1:, :] for x in features]
        B, N, C = features[0].shape
        S = int(math.sqrt(N))
        features = [x.reshape(B, S, S, C).permute(0, 3, 1, 2) for x in features]
        features = self.fuse(torch.cat(features, 1))
        # re_feat = []
        # re_feat.append(F.interpolate(features, scale_factor=4, mode='bilinear'))
        # re_feat.append(F.interpolate(features, scale_factor=2, mode='bilinear'))
        # re_feat.append(features)
        # re_feat.append(F.interpolate(features, scale_factor=1/2, mode='bilinear'))
        return features

class ViT_adapter(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, depth=None, **kwargs):
        super().__init__(embed_layer=PatchEmbed, **kwargs)
        del self.head
        self.depth=depth

        self.fuse = nn.Conv2d(768*4, 768, kernel_size=1)
        self.use_extra_extractor=True
        self.interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
        self.level_embed = nn.Parameter(torch.zeros(3, 768))
        self.spm = SpatialPriorModule(inplanes=64, embed_dim=768, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=768, num_heads=6, n_points=4,
                             init_values=0., drop_path=0.,
                             norm_layer=nn.LayerNorm, with_cffn=True,
                             cffn_ratio=0.25, deform_ratio=1.0,
                             extra_extractor=((True if i == len(self.interaction_indexes) - 1
                                               else False) and self.use_extra_extractor),
                             with_cp=False)
            for i in range(len(self.interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(768, 768, 2, 2)
        self.norm1 = nn.SyncBatchNorm(768)
        self.norm2 = nn.SyncBatchNorm(768)
        self.norm3 = nn.SyncBatchNorm(768)
        self.norm4 = nn.SyncBatchNorm(768)
        self.add_vit_feature = True

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_features(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.pos_embed
        cls_tokens = pos_embed[:, :1, :]
        pos_tokens = pos_embed[:, 1:, :]
        pos_tokens = pos_tokens.reshape(1, *self.patch_embed.grid_size, -1).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens,
            size=(H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]),
            mode='bicubic', align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat((cls_tokens, pos_tokens), dim=1)

        x = self.pos_drop(x + pos_embed)
        dim = x.shape[-1]
        H = int(math.sqrt(x.shape[1]-1))
        features=[]
        outs = list()
        x = x[:,1:,:]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1], deform_inputs1, deform_inputs2, H, H)
            outs.append(x.transpose(1, 2).view(B, dim, H, H).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(B, dim, H * 2, H * 2).contiguous()
        c3 = c3.transpose(1, 2).view(B, dim, H, H).contiguous()
        c4 = c4.transpose(1, 2).view(B, dim, H // 2, H // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
        # for i in range(self.depth):
        #     x = self.blocks[i](x)
        #     if i in [2,5,8,11]:           
        #         features.append(self.norm(x))
        # features[-1] = self.norm(features[-1])   
        return outs#features


    def forward(self, x, strides=[4, 8, 16, 32]):

        features = self.forward_features(x)

        # features = [x[:, 1:, :] for x in features]
        # B, N, C = features[0].shape
        # S = int(math.sqrt(N))
        # features = [x.reshape(B, S, S, C).permute(0, 3, 1, 2) for x in features]

        # features = self.fuse(torch.cat(features, 1))
        return features



def vit_base_patch16_224(**kwargs):
    model = ViT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.1,
        **kwargs
    )
    # pretrain = 'pretrain/mae_pretrain_vit_base.pth'
    # checkpoint = torch.load(pretrain, map_location='cpu')

    # print("Load pre-trained checkpoint from: %s" % pretrain)
    # if 'model' in checkpoint:
    #     checkpoint = checkpoint['model']

    # # interpolate position embedding
    # interpolate_pos_embed(model, checkpoint)

    # msg = model.load_state_dict(checkpoint, strict=False)
    # print(msg)

    # model.load_pretrained('pretrain/1k_vit_base.npz')
    return model
    
# class Simple_fuse(nn.Module):
#     def __init__(self, feature_channels, out):
#         super().__init__()

#         self.c0_down = nn.Conv2d(feature_channels[0], out, kernel_size=1, stride=1, padding=0)
#         self.c1_down = nn.Conv2d(feature_channels[1], out, kernel_size=1, stride=1, padding=0)
#         self.c2_down = nn.Conv2d(feature_channels[2], out, kernel_size=1, stride=1, padding=0)
#         self.c3_down = nn.Conv2d(feature_channels[3], out, kernel_size=1, stride=1, padding=0)
#         self.c4_down = nn.Conv2d(feature_channels[4], out, kernel_size=1, stride=1, padding=0)


#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         # assert len(xs) == 5
#         c0, c1, c2, c3, c4 = xs
#         c0 = self.c0_down(c0)
#         c1 = self.c1_down(c1)
#         c2 = self.c2_down(c2)
#         c3 = self.c3_down(c3)
#         c4 = self.c4_down(c4)
#         # c5 = self.c5_down(c5)        

#         return [c0, c1, c2, c3, c4]
    
class Simple_fuse(nn.Module):
    def __init__(self, feature_channels, out):
        super().__init__()

        self.c0_down = nn.Conv2d(feature_channels[0], out, kernel_size=1, stride=1, padding=0)
        self.c1_down = nn.Conv2d(feature_channels[1], out, kernel_size=1, stride=1, padding=0)
        self.c2_down = nn.Conv2d(feature_channels[2], out, kernel_size=1, stride=1, padding=0)
        self.c3_down = nn.Conv2d(feature_channels[3], out, kernel_size=1, stride=1, padding=0)


    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        # assert len(xs) == 5
        c0, c1, c2, c3 = xs
        c0 = self.c0_down(c0)
        c1 = self.c1_down(c1)
        c2 = self.c2_down(c2)
        c3 = self.c3_down(c3)      

        return [c0, c1, c2, c3]

class SimpleHead(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes, out=256):
        super(SimpleHead, self).__init__()

        feature_channels = [256,512,1024,2048]#[768,768,768,768]#
        # self.fuse=nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.fuse = Simple_fuse(feature_channels, out)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d((len(feature_channels)-1)*out, out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(out, num_classes, kernel_size=3, padding=1)

    def forward(self, features):
        features = self.fuse(features[:-1])#)#
        P = []
        
        P.extend([up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))])     
        decode_size = P[-1].size()[2:]
        P[:-1] = [F.interpolate(feature, size=decode_size, mode='bilinear', align_corners=True) for feature in P[:-1]] # 变一样大

        x = self.conv_fusion(torch.cat((P), dim=1))
        img_size = [i*4 for i in decode_size]
        x = self.head(x)
        x = F.interpolate(x, size=img_size, mode='bilinear')   
  

        return x
    
# class SimpleHead(nn.Module):
#     # Implementing only the object path
#     def __init__(self, num_classes, out=256):
#         super(SimpleHead, self).__init__()

#         feature_channels = [256,512,1024,2048,768]#[768,768,768,768]#
#         # self.fuse=nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
#         self.fuse = Simple_fuse(feature_channels, out)
#         self.conv_fusion = nn.Sequential(
#             nn.Conv2d((len(feature_channels)-1)*out, out, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out),
#             nn.ReLU(inplace=True)
#         )
#         self.head = nn.Conv2d(out, num_classes, kernel_size=3, padding=1)

#     def forward(self, features):
#         features = self.fuse(features)
#         P = []
        
#         P.append(features[-1])
#         P.extend([up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)-1))])     
#         decode_size = P[-1].size()[2:]
#         P[:-1] = [F.interpolate(feature, size=decode_size, mode='bicubic', align_corners=True) for feature in P[:-1]] # 变一样大

#         x = self.conv_fusion(torch.cat((P), dim=1))
#         img_size = [i*4 for i in decode_size]
#         x = self.head(x)
#         x = F.interpolate(x, size=img_size, mode='bicubic')   
  

#         return x


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bicubic', align_corners=True) + y

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )     
            
    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts
 
    
class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 3, 6], num_classes=1):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out
 
class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=512):# channels=2048
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)
        
        # self.vit_fuse = nn.Sequential(
        #     nn.Conv2d(768, out_channels, 1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(1024, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(512, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )    
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(256, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ) 
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)
        self.cls = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, input_fpn):
        vit_fpn = input_fpn[-1]
        # input_fpn = input_fpn[:-1]
        x1 = self.PPMHead(input_fpn[-1])
        
        # x = x1
        x = F.interpolate(x1, size=(x1.size(2)*2, x1.size(3)*2),mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)
        
        x = F.interpolate(x2, size=(x2.size(2)*2, x2.size(3)*2),mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)  
 
        x = F.interpolate(x3, size=(x3.size(2)*2, x3.size(3)*2),mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)
 
        x1 = F.interpolate(x1, x4.size()[-2:],mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:],mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:],mode='bilinear', align_corners=True)
        # x5 = F.interpolate(self.vit_fuse(vit_f), x4.size()[-2:],mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))
        # x = self.fuse_r_v(torch.cat([x, x5], 1))
        x = self.cls(x)
        s = x4.shape[-1]*4
        x = F.interpolate(x, size=(s,s), mode='bilinear')  
        return x

class vit_fuse(nn.Module):
    def __init__(self, feature_channels, out):
        super().__init__()

        self.c0_down = nn.Conv2d(feature_channels[0], out, kernel_size=1, stride=1, padding=0)
        self.c1_down = nn.Conv2d(feature_channels[1], out, kernel_size=1, stride=1, padding=0)
        self.c2_down = nn.Conv2d(feature_channels[2], out, kernel_size=1, stride=1, padding=0)
        self.c3_down = nn.Conv2d(feature_channels[3], out, kernel_size=1, stride=1, padding=0)
        self.c4_down = nn.Conv2d(feature_channels[4], out, kernel_size=1, stride=1, padding=0)


    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        # assert len(xs) == 5
        c0, c1, c2, c3, c4 = xs
        c0 = self.c0_down(c0)
        c1 = self.c1_down(c1)
        c2 = self.c2_down(c2)
        c3 = self.c3_down(c3)
        c4 = self.c4_down(c4)
        # c5 = self.c5_down(c5)        

        return [c0, c1, c2, c3, c4]


    



class vitHead(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes, out=256):
        super(vitHead, self).__init__()

        self.fuse_1 = nn.Sequential(
            nn.Conv2d(768, out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
        self.fuse_2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, features):
        feat = features[-1][:,1:,:].permute(0,2,1)
        feat = feat.reshape(feat.shape[0], 768, 24, 24)
        # feat = features.permute(0,2,1)
        # feat = feat.reshape(feat.shape[0], 1024, 12, 12)
        features = self.fuse_2(self.fuse_1(feat))
        x = self.head(features)
        x = F.interpolate(x, size=(384,384), mode='bilinear')   
  
        # feature = self.fuse(features[0])
        # feature = self.head(feature)
        # x = F.interpolate(feature, size=(384,384), mode='bilinear')

        return x
    
@export
class PretrainInitHook(Hook):
    """Init with pretrained model"""
    priority = 'NORMAL'

    def __init__(self):
        pass
    def before_train(self, runner):
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        # model.space_encoder = resnet50().cuda()
        # model.higher_encoder = vit_base_patch16_224().cuda()

        # Load pretrain for resnet50
        # pretrain = 'pretrain/resnet50_pretrain.pth' 
        # checkpoint = torch.load(pretrain, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % pretrain)
        # if 'model' in checkpoint:
        #     checkpoint = checkpoint['model']
        # msg = model.space_encoder.load_state_dict(checkpoint, strict=False)
        # print(msg)
        # msg = model.higher_encoder.load_state_dict(checkpoint, strict=False)
        # print(msg)

        # Load pretrain for vit_base
        # pretrain = 'pretrain/deit_base_distilled_patch16_384-d0272ac0.pth'#mae_pretrain_vit_base.pth'
        # checkpoint = torch.load(pretrain, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % pretrain)
        # if 'model' in checkpoint:
        #     checkpoint = checkpoint['model']
        # interpolate_pos_embed(model.higher_encoder, checkpoint)
        # msg = model.higher_encoder.load_state_dict(checkpoint, strict=False)
        # print(msg)
        # interpolate_pos_embed(model.space_encoder, checkpoint)
        # msg = model.space_encoder.load_state_dict(checkpoint, strict=False)
        # print(msg)

        # Load pretrain for swin
        # pretrain = 'pretrain/swin_base_patch4_window12_384_22k.pth'
        # checkpoint = torch.load(pretrain, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % pretrain)
        # if 'model' in checkpoint:
        #     checkpoint = checkpoint['model']
        # new_state_dict = {}
        # # import re
        # # for k, v in checkpoint.items():
        # #     k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
        # #     k = k.replace('head.', 'head.fc.')
        # #     new_state_dict[k] = v
        # msg = model.space_encoder.load_state_dict(checkpoint, strict=False)
        # print(msg)


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

        

        # # load pretrain for sam
        # sam_checkpoint = "pretrain/sam_vit_b_01ec64.pth"
        # checkpoint = torch.load(sam_checkpoint, map_location='cpu')
        # print("Load pre-trained checkpoint for sam from: %s" % sam_checkpoint)
        # msg = model.sam.load_state_dict(checkpoint, strict=False)
        # print(msg)

    # def before_val(self, runner):
    #     model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model

    #     # Load checkpoint of hitnet 
    #     pretrain = 'output/test/epoch_80.pth'
    #     checkpoint = torch.load(pretrain, map_location='cpu')
    #     print("Load pre-trained checkpoint from: %s" % pretrain)
    #     if 'model' in checkpoint:
    #         checkpoint = checkpoint['model']
    #     msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     print(msg)

    #     # Load checkpoint of depth head
    #     pretrain_depth_head = 'output/nyu/epoch_20.pth'
    #     checkpoint = torch.load(pretrain_depth_head, map_location='cpu')
    #     print("Load pre-trained checkpoint from: %s" % pretrain_depth_head)
    #     if 'model' in checkpoint:
    #         checkpoint = checkpoint['model']
    #     msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     print(msg)

    #     # Load checkpoint for eval after finetuned sam
    #     pretrain = 'output/hitnet/epoch_20.pth'       
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


        # sam_checkpoint = "pretrain/sam_vit_h_4b8939.pth"
        # checkpoint = torch.load(sam_checkpoint, map_location='cpu')
        # print("Load pre-trained checkpoint for sam from: %s" % sam_checkpoint)
        # msg = model.sam.load_state_dict(checkpoint, strict=False)
        # print(msg)



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
        pvt = self.backbone(x, pred_normal)
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
        return stage_loss, prediction2_8



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

class Depth_prompt(nn.Module):
    def __init__(self, scale_factor, input_dim, embed_dim, depth):
        super(Depth_prompt, self).__init__()
        self.scale_factor = 2#scale_factor
        self.embed_dim = embed_dim
        self.depth = depth
        self.input_dim = input_dim

        self.shared_mlp = nn.Linear(self.input_dim//self.scale_factor, self.embed_dim)
        # self.embedding_generator = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(1, self.input_dim//16),#1->48
        #     nn.GELU(),
        #     nn.Linear(self.input_dim//16, self.input_dim//8),#48->96
        #     nn.GELU(),
        #     nn.Linear(self.input_dim//8, self.input_dim//self.scale_factor),#96->192
        # )
        self.embedding_generator = nn.Sequential(
            # nn.GELU(),
            nn.Linear(1, self.input_dim),
            nn.Linear(self.input_dim, self.input_dim//self.scale_factor)
            # nn.Linear(1, self.embed_dim)
        )
        # self.embedding_generator = nn.Linear(self.input_dim, self.input_dim//self.scale_factor)
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.input_dim//self.scale_factor, self.input_dim//self.scale_factor),
                nn.GELU(),
            )
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

    def init_embeddings(self, x):
        x = x.permute(0,3,1,2).contiguous()
        N, C, H, W = x.shape
        x = x.reshape(N, C, H*W).permute(0, 2, 1)
        return self.embedding_generator(x)


    def forward(self, depth):
        N, C, H, W = depth.shape
        depth_feature = depth.view(N, C, H*W).permute(0, 2, 1)
        depth_feature = self.embedding_generator(depth_feature)
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            prompt = lightweight_mlp(depth_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts  

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

        # depth prompt
        self.dino_dim = 768
        self.scale_factor = 4
        self.depth_generator = nn.ModuleList([
            Depth_prompt(self.scale_factor, self.dino_dim, embed_dims[0], self.depths[0]),
            Depth_prompt(self.scale_factor, self.dino_dim, embed_dims[1], self.depths[1]),
            Depth_prompt(self.scale_factor, self.dino_dim, embed_dims[2], self.depths[2]),
            Depth_prompt(self.scale_factor, self.dino_dim, embed_dims[3], self.depths[3]),
            ])

        self.cross = nn.ModuleList([
            WindowFusion(64),
            WindowFusion(128),
            WindowFusion(320),
            WindowFusion(512),
            ])

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



    def forward_features(self, x, pred_normal):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)

        depth = F.interpolate(pred_normal, size=(H,W), mode='bilinear')
        depth = self.depth_generator[0](depth)
        # for i, d in enumerate(depth):
        #     b,n,c = d.shape
        #     h = int(math.sqrt(n))
        #     d = d.permute(0,2,1).reshape(b,c,h,h)
        #     d = F.interpolate(d, size=(H,W), mode='bilinear')
        #     depth[i] = d.permute(0,2,3,1).reshape(b,H*W,c)

        for i, blk in enumerate(self.block1):
            # if True:#i == self.depths[0]-1:
            #     fused = self.cross[0](depth[i].reshape(x.shape), x)[0]
            #     B,C,H,W = fused.shape
            #     fused = fused.reshape(B,C,H*W).permute(0,2,1)
                
            # x = blk(fused, H, W)
            x = blk(x+depth[i].reshape(x.shape), H, W) #10, 176^2, 64
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)

        depth = F.interpolate(pred_normal, size=(H,W), mode='bilinear')
        depth = self.depth_generator[1](depth)
        # for i, d in enumerate(depth):
        #     b,n,c = d.shape
        #     h = int(math.sqrt(n))
        #     d = d.permute(0,2,1).reshape(b,c,h,h)
        #     d = F.interpolate(d, size=(H,W), mode='bilinear')
        #     depth[i] = d.permute(0,2,3,1).reshape(b,H*W,c)
        
        for i, blk in enumerate(self.block2):
            #     fused = self.cross[1](depth[i].reshape(x.shape), x)[0]
            #     B,C,H,W = fused.shape
            #     fused = fused.reshape(B,C,H*W).permute(0,2,1)                
            # x = blk(fused, H, W)
            x = blk(x+depth[i].reshape(x.shape), H, W) #10, 88^2, 128
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)

        depth = F.interpolate(pred_normal, size=(H,W), mode='bilinear')
        depth = self.depth_generator[2](depth)
        # for i, d in enumerate(depth):
        #     b,n,c = d.shape
        #     h = int(math.sqrt(n))
        #     d = d.permute(0,2,1).reshape(b,c,h,h)
        #     d = F.interpolate(d, size=(H,W), mode='bilinear')
        #     depth[i] = d.permute(0,2,3,1).reshape(b,H*W,c)
        
        for i, blk in enumerate(self.block3):
            #     fused = self.cross[2](depth[i].reshape(x.shape), x)[0]
            #     B,C,H,W = fused.shape
            #     fused = fused.reshape(B,C,H*W).permute(0,2,1)              
            # x = blk(fused, H, W)
            x = blk(x+depth[i].reshape(x.shape), H, W) #10, 44^2, 320
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)

        depth = F.interpolate(pred_normal, size=(H,W), mode='bilinear')
        depth = self.depth_generator[3](depth)
        # for i, d in enumerate(depth):
        #     b,n,c = d.shape
        #     h = int(math.sqrt(n))
        #     d = d.permute(0,2,1).reshape(b,c,h,h)
        #     d = F.interpolate(d, size=(H,W), mode='bilinear')
        #     depth[i] = d.permute(0,2,3,1).reshape(b,H*W,c)
        
        for i, blk in enumerate(self.block4):
            #     fused = self.cross[3](depth[i].reshape(x.shape), x)[0]
            #     B,C,H,W = fused.shape
            #     fused = fused.reshape(B,C,H*W).permute(0,2,1)   
            # x = blk(fused, H, W)
            x = blk(x+depth[i].reshape(x.shape), H, W) #10, 22^2, 512
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs

        # return x.mean(dim=1)

    def forward(self, x, depth):
        x = self.forward_features(x, depth)
        # x = self.head(x)

        return x


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



        return x+identity+identity_y, x.sigmoid()#bias



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






















# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------



# import MultiScaleDeformableAttention as MSDA

# from torch.autograd import Function
# from torch.autograd.function import once_differentiable
# from torch.cuda.amp import custom_bwd, custom_fwd
# import math
# import warnings

# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.nn.init import constant_, xavier_uniform_


# def _is_power_of_2(n):
#     if (not isinstance(n, int)) or (n < 0):
#         raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
#     return (n & (n - 1) == 0) and n != 0

# class InteractionBlock(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, norm_layer=nn.LayerNorm,
#                  drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
#                  deform_ratio=1.0, extra_extractor=False, with_cp=False):
#         super().__init__()

#         self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
#                                  n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
#                                  with_cp=with_cp)
#         self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
#                                    norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
#                                    cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
#         if extra_extractor:
#             self.extra_extractors = nn.Sequential(*[
#                 Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
#                           with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
#                           drop=drop, drop_path=drop_path, with_cp=with_cp)
#                 for _ in range(2)
#             ])
#         else:
#             self.extra_extractors = None

#     def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
#         x = self.injector(query=x, reference_points=deform_inputs1[0],
#                           feat=c, spatial_shapes=deform_inputs1[1],
#                           level_start_index=deform_inputs1[2])
#         for idx, blk in enumerate(blocks):
#             x = blk(x)
#         c = self.extractor(query=c, reference_points=deform_inputs2[0],
#                            feat=x, spatial_shapes=deform_inputs2[1],
#                            level_start_index=deform_inputs2[2], H=H, W=W)
#         if self.extra_extractors is not None:
#             for extractor in self.extra_extractors:
#                 c = extractor(query=c, reference_points=deform_inputs2[0],
#                               feat=x, spatial_shapes=deform_inputs2[1],
#                               level_start_index=deform_inputs2[2], H=H, W=W)
#         return x, c


# class Extractor(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
#                  with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
#                  norm_layer=nn.LayerNorm, with_cp=False):
#         super().__init__()
#         self.query_norm = norm_layer(dim)
#         self.feat_norm = norm_layer(dim)
#         self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
#                                  n_points=n_points, ratio=deform_ratio)
#         self.with_cffn = with_cffn
#         self.with_cp = with_cp
#         if with_cffn:
#             self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
#             self.ffn_norm = norm_layer(dim)
#             self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
#         def _inner_forward(query, feat):

#             attn = self.attn(self.query_norm(query), reference_points,
#                              self.feat_norm(feat), spatial_shapes,
#                              level_start_index, None)
#             query = query + attn
    
#             if self.with_cffn:
#                 query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
#             return query
        
#         if self.with_cp and query.requires_grad:
#             query = cp.checkpoint(_inner_forward, query, feat)
#         else:
#             query = _inner_forward(query, feat)
            
#         return query

# class ConvFFN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None,
#                  act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         n = N // 21
#         x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
#         x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
#         x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
#         x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
#         x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
#         x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
#         x = torch.cat([x1, x2, x3], dim=1)
#         return x


# class MSDeformAttn(nn.Module):
#     'deformable ops'
#     def __init__(self, d_model:int=256, n_levels:int=4, n_heads:int=8, n_points:int=4, ratio:float=1.0):
#         """Multi-Scale Deformable Attention Module.

#         :param d_model      hidden dimension
#         :param n_levels     number of feature levels
#         :param n_heads      number of attention heads
#         :param n_points     number of sampling points per attention head per feature level
#         """
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, '
#                              'but got {} and {}'.format(d_model, n_heads))
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2
#         # which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_head):
#             warnings.warn(
#                 "You'd better set d_model in MSDeformAttn to make "
#                 'the dimension of each attention head a power of 2 '
#                 'which is more efficient in our CUDA implementation.')

#         self.im2col_step = 64

#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads
#         self.n_points = n_points
#         self.ratio = ratio
#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
#         self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
#         self.value_proj = nn.Linear(d_model, int(d_model * ratio))
#         self.output_proj = nn.Linear(int(d_model * ratio), d_model)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         constant_(self.sampling_offsets.weight.data, 0.)
#         thetas = torch.arange(
#             self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
#                          self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
#         for i in range(self.n_points):
#             grid_init[:, :, i, :] *= i + 1

#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.)
#         constant_(self.attention_weights.bias.data, 0.)
#         xavier_uniform_(self.value_proj.weight.data)
#         constant_(self.value_proj.bias.data, 0.)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.)

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
#                 input_level_start_index, input_padding_mask=None):
#         """
#         :param query                       (N, Length_{query}, C)
#         :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
#                                         or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
#         :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
#         :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#         :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
#         :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

#         :return output                     (N, Length_{query}, C)
#         """

#         N, Len_q, _ = query.shape
#         N, Len_in, _ = input_flatten.shape
#         assert (input_spatial_shapes[:, 0] *
#                 input_spatial_shapes[:, 1]).sum() == Len_in

#         value = self.value_proj(input_flatten)
#         if input_padding_mask is not None:
#             value = value.masked_fill(input_padding_mask[..., None], float(0))

#         value = value.view(N, Len_in, self.n_heads,
#                            int(self.ratio * self.d_model) // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(
#             N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
#         attention_weights = self.attention_weights(query).view(
#             N, Len_q, self.n_heads, self.n_levels * self.n_points)
#         attention_weights = F.softmax(attention_weights, -1).\
#             view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack(
#                 [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_points[:, :, None, :, None, :] \
#                                  + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = reference_points[:, :, None, :, None, :2] \
#                                  + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
#         else:
#             raise ValueError(
#                 'Last dim of reference_points must be 2 or 4, but get {} instead.'
#                 .format(reference_points.shape[-1]))
#         output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index,
#                                             sampling_locations, attention_weights, self.im2col_step)
#         output = self.output_proj(output)
#         return output

# # ------------------------------------------------------------------------------------------------
# # Deformable DETR
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------------------------------
# # Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# # ------------------------------------------------------------------------------------------------




# class MSDeformAttnFunction(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index,
#                 sampling_locations, attention_weights, im2col_step):
#         ctx.im2col_step = im2col_step
#         output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
#                                              value_level_start_index,
#                                              sampling_locations,
#                                              attention_weights,
#                                              ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes,
#                               value_level_start_index, sampling_locations,
#                               attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     @custom_bwd
#     def backward(ctx, grad_output):
#         value, value_spatial_shapes, value_level_start_index, \
#         sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value, grad_sampling_loc, grad_attn_weight = \
#             MSDA.ms_deform_attn_backward(
#                 value, value_spatial_shapes, value_level_start_index,
#                 sampling_locations, attention_weights, grad_output, ctx.im2col_step)

#         return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


# def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
#                                 sampling_locations, attention_weights):
#     # for debug and test only,
#     # need to use cuda version instead
#     N_, S_, M_, D_ = value.shape
#     _, Lq_, M_, L_, P_, _ = sampling_locations.shape
#     value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
#     sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#     for lid_, (H_, W_) in enumerate(value_spatial_shapes):
#         # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
#         value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
#         # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
#         sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
#         # N_*M_, D_, Lq_, P_
#         sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear',
#                                           padding_mode='zeros', align_corners=False)
#         sampling_value_list.append(sampling_value_l_)
#     # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
#     attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
#     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
#               attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
#     return output.transpose(1, 2).contiguous()

# def get_reference_points(spatial_shapes, device):
#     reference_points_list = []
#     for lvl, (H_, W_) in enumerate(spatial_shapes):
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#             torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
#         ref_y = ref_y.reshape(-1)[None] / H_
#         ref_x = ref_x.reshape(-1)[None] / W_
#         ref = torch.stack((ref_x, ref_y), -1)
#         reference_points_list.append(ref)
#     reference_points = torch.cat(reference_points_list, 1)
#     reference_points = reference_points[:, :, None]
#     return reference_points


# def deform_inputs(x):
#     bs, c, h, w = x.shape
#     spatial_shapes = torch.as_tensor([(h // 8, w // 8),
#                                       (h // 16, w // 16),
#                                       (h // 32, w // 32)],
#                                      dtype=torch.long, device=x.device)
#     level_start_index = torch.cat((spatial_shapes.new_zeros(
#         (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#     reference_points = get_reference_points([(h // 16, w // 16)], x.device)
#     deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
#     spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
#     level_start_index = torch.cat((spatial_shapes.new_zeros(
#         (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#     reference_points = get_reference_points([(h // 8, w // 8),
#                                              (h // 16, w // 16),
#                                              (h // 32, w // 32)], x.device)
#     deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
#     return deform_inputs1, deform_inputs2

# class SpatialPriorModule(nn.Module):
#     def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
#         super().__init__()
#         self.with_cp = with_cp

#         self.stem = nn.Sequential(*[
#             nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         ])
#         self.conv2 = nn.Sequential(*[
#             nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(2 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv3 = nn.Sequential(*[
#             nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv4 = nn.Sequential(*[
#             nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
        
#         def _inner_forward(x):
#             c1 = self.stem(x)
#             c2 = self.conv2(c1)
#             c3 = self.conv3(c2)
#             c4 = self.conv4(c3)
#             c1 = self.fc1(c1)
#             c2 = self.fc2(c2)
#             c3 = self.fc3(c3)
#             c4 = self.fc4(c4)
#             bs, dim, _, _ = c1.shape
#             # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
#             c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
#             c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
#             c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
#             return c1, c2, c3, c4
        
#         if self.with_cp and x.requires_grad:
#             outs = cp.checkpoint(_inner_forward, x)
#         else:
#             outs = _inner_forward(x)
#         return outs

# class Injector(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
#                  norm_layer=nn.LayerNorm, init_values=0., with_cp=False):
#         super().__init__()
#         self.with_cp = with_cp
#         self.query_norm = norm_layer(dim)
#         self.feat_norm = norm_layer(dim)
#         self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
#                                  n_points=n_points, ratio=deform_ratio)
#         self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

#     def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
#         def _inner_forward(query, feat):

#             attn = self.attn(self.query_norm(query), reference_points,
#                              self.feat_norm(feat), spatial_shapes,
#                              level_start_index, None)
#             return query + self.gamma * attn
        
#         if self.with_cp and query.requires_grad:
#             query = cp.checkpoint(_inner_forward, query, feat)
#         else:
#             query = _inner_forward(query, feat)
            
#         return query





# class NewCRFDepth(nn.Module):
#     """
#     Depth network based on neural window FC-CRFs architecture.
#     """
#     def __init__(self, version=None, inv_depth=False, pretrained=None, 
#                     frozen_stages=-1, min_depth=0.1, max_depth=10, **kwargs):
#         super().__init__()

#         self.inv_depth = inv_depth
#         self.with_auxiliary_head = False
#         self.with_neck = False

#         norm_cfg = dict(type='BN', requires_grad=True)
#         # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

#         window_size = 7#int(version[-2:])

#         # if version[:-2] == 'base':
#             # embed_dim = 128
#             # depths = [2, 2, 18, 2]
#             # num_heads = [4, 8, 16, 32]
#             # in_channels = [128, 256, 512, 1024]
#         # elif version[:-2] == 'large':
#         embed_dim = 192
#         depths = [2, 2, 18, 2]
#         num_heads = [6, 12, 24, 48]
#         in_channels = [192, 384, 768, 1536]
#         # elif version[:-2] == 'tiny':
#         #     embed_dim = 96
#         #     depths = [2, 2, 6, 2]
#         #     num_heads = [3, 6, 12, 24]
#         #     in_channels = [96, 192, 384, 768]

#         # embed_dim = 768
#         # depths = [3, 3, 3, 3]
#         # num_heads = [12, 12, 12, 12]
#         # in_channels = [768, 768, 768, 768]

#         backbone_cfg = dict(
#             embed_dim=embed_dim,
#             depths=depths,
#             num_heads=num_heads,
#             window_size=window_size,
#             ape=False,
#             drop_path_rate=0.3,
#             patch_norm=True,
#             use_checkpoint=False,
#             frozen_stages=frozen_stages
#         )

#         embed_dim = 512
#         decoder_cfg = dict(
#             in_channels=in_channels,
#             in_index=[0, 1, 2, 3],
#             pool_scales=(1, 2, 3, 6),
#             channels=embed_dim,
#             dropout_ratio=0.0,
#             num_classes=32,
#             norm_cfg=norm_cfg,
#             align_corners=False
#         )

#         self.backbone = SwinTransformer(**backbone_cfg)
#         v_dim = decoder_cfg['num_classes']*4
#         win = 7
#         crf_dims = [128, 256, 512, 1024]
#         v_dims = [64, 128, 256, embed_dim]
#         self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
#         self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
#         self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
#         self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

#         self.decoder = PSP(**decoder_cfg)
#         self.disp_head1 = DispHead(input_dim=crf_dims[0])

#         self.up_mode = 'bilinear'
#         if self.up_mode == 'mask':
#             self.mask_head = nn.Sequential(
#                 nn.Conv2d(crf_dims[0], 64, 3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 16*9, 1, padding=0))

#         self.min_depth = min_depth
#         self.max_depth = max_depth

#         self.init_weights(pretrained=pretrained)

#     def init_weights(self, pretrained=None):
#         """Initialize the weights in backbone and heads.

#         Args:
#             pretrained (str, optional): Path to pre-trained weights.
#                 Defaults to None.
#         """
#         print(f'== Load encoder backbone from: {pretrained}')
#         # self.backbone.init_weights(pretrained=pretrained)
#         self.decoder.init_weights()
#         if self.with_auxiliary_head:
#             if isinstance(self.auxiliary_head, nn.ModuleList):
#                 for aux_head in self.auxiliary_head:
#                     aux_head.init_weights()
#             else:
#                 self.auxiliary_head.init_weights()

#     def upsample_mask(self, disp, mask):
#         """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
#         N, _, H, W = disp.shape
#         mask = mask.view(N, 1, 9, 4, 4, H, W)
#         mask = torch.softmax(mask, dim=2)

#         up_disp = F.unfold(disp, kernel_size=3, padding=1)
#         up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

#         up_disp = torch.sum(mask * up_disp, dim=2)
#         up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
#         return up_disp.reshape(N, 1, 4*H, 4*W)

#     def forward(self, imgs):#feats):
#         feats = self.backbone(imgs)
#         if self.with_neck:
#             feats = self.neck(feats)
#         ppm_out = self.decoder(feats)

#         e3 = self.crf3(feats[3], ppm_out)
#         e3 = nn.PixelShuffle(2)(e3)
#         # feats[2] = F.interpolate(feats[2], scale_factor=2, mode='bilinear')
#         e2 = self.crf2(feats[2], e3)
#         e2 = nn.PixelShuffle(2)(e2)
#         # feats[1] = F.interpolate(feats[1], scale_factor=4, mode='bilinear')
#         e1 = self.crf1(feats[1], e2)
#         e1 = nn.PixelShuffle(2)(e1)
#         # feats[0] = F.interpolate(feats[0], scale_factor=8, mode='bilinear')
#         e0 = self.crf0(feats[0], e1)

#         if self.up_mode == 'mask':
#             mask = self.mask_head(e0)
#             d1 = self.disp_head1(e0, 1)
#             d1 = self.upsample_mask(d1, mask)
#         else:
#             d1 = self.disp_head1(e0, 4)
#             # d1 = self.disp_head1(e0, 1)

#         depth = d1 * self.max_depth
#         return depth


# class DispHead(nn.Module):
#     def __init__(self, input_dim=100):
#         super(DispHead, self).__init__()
#         # self.norm1 = nn.BatchNorm2d(input_dim)
#         self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
#         # self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, scale):
#         # x = self.relu(self.norm1(x))
#         x = self.sigmoid(self.conv1(x))
#         if scale > 1:
#             x = upsample(x, scale_factor=scale)
#         return x


# class DispUnpack(nn.Module):
#     def __init__(self, input_dim=100, hidden_dim=128):
#         super(DispUnpack, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
#         self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#         self.pixel_shuffle = nn.PixelShuffle(4)

#     def forward(self, x, output_size):
#         x = self.relu(self.conv1(x))
#         x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
#         # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
#         x = self.pixel_shuffle(x)

#         return x


# def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
#     """Upsample input tensor by a factor of 2
#     """
#     return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


# import torch.utils.checkpoint as checkpoint
# import numpy as np
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# class this_Mlp(nn.Module):
#     """ Multilayer perceptron."""

#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows


# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x


# class WindowAttention(nn.Module):
#     """ Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, v_dim, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(v_dim, v_dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, v, mask=None):
#         """ Forward function.

#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qk = self.qk(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)
        
#         # assert self.dim % v.shape[-1] == 0, "self.dim % v.shape[-1] != 0"
#         # repeat_num = self.dim // v.shape[-1]
#         # v = v.view(B_, N, self.num_heads // repeat_num, -1).transpose(1, 2).repeat(1, repeat_num, 1, 1)

#         assert self.dim == v.shape[-1], "self.dim != v.shape[-1]"
#         v = v.view(B_, N, self.num_heads, -1).transpose(1, 2)
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class CRFBlock(nn.Module):
#     """ CRF Block.

#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, dim, num_heads, v_dim, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.v_dim = v_dim
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, v_dim=v_dim,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(v_dim)
#         mlp_hidden_dim = int(v_dim * mlp_ratio)
#         self.mlp = this_Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.H = None
#         self.W = None

#     def forward(self, x, v, mask_matrix):
#         """ Forward function.

#         Args:
#             x: Input feature, tensor size (B, H*W, C).
#             H, W: Spatial resolution of the input feature.
#             mask_matrix: Attention mask for cyclic shift.
#         """
#         B, L, C = x.shape
#         H, W = self.H, self.W
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)

#         # pad feature maps to multiples of window size
#         pad_l = pad_t = 0
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
#         v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
#         _, Hp, Wp, _ = x.shape

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#             shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#             attn_mask = mask_matrix
#         else:
#             shifted_x = x
#             shifted_v = v
#             attn_mask = None

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
#         v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
#         v_windows = v_windows.view(-1, self.window_size * self.window_size, v_windows.shape[-1])  # nW*B, window_size*window_size, C
        
#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, v_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.v_dim)
#         shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x

#         if pad_r > 0 or pad_b > 0:
#             x = x[:, :H, :W, :].contiguous()

#         x = x.view(B, H * W, self.v_dim)

#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x


# class BasicCRFLayer(nn.Module):
#     """ A basic NeWCRFs layer for one stage.

#     Args:
#         dim (int): Number of feature channels
#         depth (int): Depths of this stage.
#         num_heads (int): Number of attention head.
#         window_size (int): Local window size. Default: 7.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(self,
#                  dim,
#                  depth,
#                  num_heads,
#                  v_dim,
#                  window_size=7,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  norm_layer=nn.LayerNorm,
#                  downsample=None,
#                  use_checkpoint=False):
#         super().__init__()
#         self.window_size = window_size
#         self.shift_size = window_size // 2
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint

#         # build blocks
#         self.blocks = nn.ModuleList([
#             CRFBlock(
#                 dim=dim,
#                 num_heads=num_heads,
#                 v_dim=v_dim,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else window_size // 2,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer)
#             for i in range(depth)])

#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None

#     def forward(self, x, v, H, W):
#         """ Forward function.

#         Args:
#             x: Input feature, tensor size (B, H*W, C).
#             H, W: Spatial resolution of the input feature.
#         """

#         # calculate attention mask for SW-MSA
#         Hp = int(np.ceil(H / self.window_size)) * self.window_size
#         Wp = int(np.ceil(W / self.window_size)) * self.window_size
#         img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1

#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

#         for blk in self.blocks:
#             blk.H, blk.W = H, W
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, attn_mask)
#             else:
#                 x = blk(x, v, attn_mask)
#         if self.downsample is not None:
#             x_down = self.downsample(x, H, W)
#             Wh, Ww = (H + 1) // 2, (W + 1) // 2
#             return x, H, W, x_down, Wh, Ww
#         else:
#             return x, H, W, x, H, W


# class NewCRF(nn.Module):
#     def __init__(self,
#                  input_dim=96,
#                  embed_dim=96,
#                  v_dim=64,
#                  window_size=7,
#                  num_heads=4,
#                  depth=2,
#                  patch_size=4,
#                  in_chans=3,
#                  norm_layer=nn.LayerNorm,
#                  patch_norm=True):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.patch_norm = patch_norm
        
#         if input_dim != embed_dim:
#             self.proj_x = nn.Conv2d(input_dim, embed_dim, 3, padding=1)
#         else:
#             self.proj_x = None

#         if v_dim != embed_dim:
#             self.proj_v = nn.Conv2d(v_dim, embed_dim, 3, padding=1)
#         elif embed_dim % v_dim == 0:
#             self.proj_v = None

#         # For now, v_dim need to be equal to embed_dim, because the output of window-attn is the input of shift-window-attn
#         v_dim = embed_dim
#         assert v_dim == embed_dim

#         self.crf_layer = BasicCRFLayer(
#                 dim=embed_dim,
#                 depth=depth,
#                 num_heads=num_heads,
#                 v_dim=v_dim,
#                 window_size=window_size,
#                 mlp_ratio=4.,
#                 qkv_bias=True,
#                 qk_scale=None,
#                 drop=0.,
#                 attn_drop=0.,
#                 drop_path=0.,
#                 norm_layer=norm_layer,
#                 downsample=None,
#                 use_checkpoint=False)

#         layer = norm_layer(embed_dim)
#         layer_name = 'norm_crf'
#         self.add_module(layer_name, layer)


#     def forward(self, x, v):
#         if self.proj_x is not None:
#             x = self.proj_x(x)
#         if self.proj_v is not None:
#             v = self.proj_v(v)

#         Wh, Ww = x.size(2), x.size(3)
#         x = x.flatten(2).transpose(1, 2)
#         v = v.transpose(1, 2).transpose(2, 3)
#         x_out, H, W, x, Wh, Ww = self.crf_layer(x, v, Wh, Ww)
#         norm_layer = getattr(self, f'norm_crf')
#         x_out = norm_layer(x_out)
#         out = x_out.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

#         return out

# from mmcv.cnn import ConvModule

# def normal_init(module, mean=0, std=1, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.normal_(module.weight, mean, std)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)

# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):
#     if warning:
#         if size is not None and align_corners:
#             input_h, input_w = tuple(int(x) for x in input.shape[2:])
#             output_h, output_w = tuple(int(x) for x in size)
#             if output_h > input_h or output_w > output_h:
#                 if ((output_h > 1 and output_w > 1 and input_h > 1
#                      and input_w > 1) and (output_h - 1) % (input_h - 1)
#                         and (output_w - 1) % (input_w - 1)):
#                     warnings.warn(
#                         f'When align_corners={align_corners}, '
#                         'the output would more aligned if '
#                         f'input size {(input_h, input_w)} is `x+1` and '
#                         f'out size {(output_h, output_w)} is `nx+1`')
#     if isinstance(size, torch.Size):
#         size = tuple(int(x) for x in size)
#     return F.interpolate(input, size, scale_factor, mode, align_corners)

# class PPM(nn.ModuleList):
#     """Pooling Pyramid Module used in PSPNet.

#     Args:
#         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
#             Module.
#         in_channels (int): Input channels.
#         channels (int): Channels after modules, before conv_seg.
#         conv_cfg (dict|None): Config of conv layers.
#         norm_cfg (dict|None): Config of norm layers.
#         act_cfg (dict): Config of activation layers.
#         align_corners (bool): align_corners argument of F.interpolate.
#     """

#     def __init__(self, pool_scales, in_channels, channels, norm_cfg,
#                  act_cfg, align_corners, conv_cfg=None):
#         super(PPM, self).__init__()
#         self.pool_scales = pool_scales
#         self.align_corners = align_corners
#         self.in_channels = in_channels
#         self.channels = channels
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         for pool_scale in pool_scales:
#             # == if batch size = 1, BN is not supported, change to GN
#             if pool_scale == 1: norm_cfg = dict(type='GN', requires_grad=True, num_groups=256)
#             self.append(
#                 nn.Sequential(
#                     nn.AdaptiveAvgPool2d(pool_scale),
#                     ConvModule(
#                         self.in_channels,
#                         self.channels,
#                         1,
#                         conv_cfg=self.conv_cfg,
#                         norm_cfg=norm_cfg,
#                         act_cfg=self.act_cfg)))

#     def forward(self, x):
#         """Forward function."""
#         ppm_outs = []
#         for ppm in self:
#             ppm_out = ppm(x)
#             upsampled_ppm_out = resize(
#                 ppm_out,
#                 size=x.size()[2:],
#                 mode='bilinear',
#                 align_corners=self.align_corners)
#             ppm_outs.append(upsampled_ppm_out)
#         return ppm_outs


# class BaseDecodeHead(nn.Module):
#     """Base class for BaseDecodeHead.

#     Args:
#         in_channels (int|Sequence[int]): Input channels.
#         channels (int): Channels after modules, before conv_seg.
#         num_classes (int): Number of classes.
#         dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
#         conv_cfg (dict|None): Config of conv layers. Default: None.
#         norm_cfg (dict|None): Config of norm layers. Default: None.
#         act_cfg (dict): Config of activation layers.
#             Default: dict(type='ReLU')
#         in_index (int|Sequence[int]): Input feature index. Default: -1
#         input_transform (str|None): Transformation type of input features.
#             Options: 'resize_concat', 'multiple_select', None.
#             'resize_concat': Multiple feature maps will be resize to the
#                 same size as first one and than concat together.
#                 Usually used in FCN head of HRNet.
#             'multiple_select': Multiple feature maps will be bundle into
#                 a list and passed into decode head.
#             None: Only one select feature map is allowed.
#             Default: None.
#         loss_decode (dict): Config of decode loss.
#             Default: dict(type='CrossEntropyLoss').
#         ignore_index (int | None): The label index to be ignored. When using
#             masked BCE loss, ignore_index should be set to None. Default: 255
#         sampler (dict|None): The config of segmentation map sampler.
#             Default: None.
#         align_corners (bool): align_corners argument of F.interpolate.
#             Default: False.
#     """

#     def __init__(self,
#                  in_channels,
#                  channels,
#                  *,
#                  num_classes,
#                  dropout_ratio=0.1,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  act_cfg=dict(type='ReLU'),
#                  in_index=-1,
#                  input_transform=None,
#                  loss_decode=dict(
#                      type='CrossEntropyLoss',
#                      use_sigmoid=False,
#                      loss_weight=1.0),
#                  ignore_index=255,
#                  sampler=None,
#                  align_corners=False):
#         super(BaseDecodeHead, self).__init__()
#         self._init_inputs(in_channels, in_index, input_transform)
#         self.channels = channels
#         self.num_classes = num_classes
#         self.dropout_ratio = dropout_ratio
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         self.in_index = in_index
#         # self.loss_decode = build_loss(loss_decode)
#         self.ignore_index = ignore_index
#         self.align_corners = align_corners
#         # if sampler is not None:
#         #     self.sampler = build_pixel_sampler(sampler, context=self)
#         # else:
#         #     self.sampler = None

#         # self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
#         # self.conv1 = nn.Conv2d(channels, num_classes, 3, padding=1)
#         if dropout_ratio > 0:
#             self.dropout = nn.Dropout2d(dropout_ratio)
#         else:
#             self.dropout = None
#         self.fp16_enabled = False

#     def extra_repr(self):
#         """Extra repr."""
#         s = f'input_transform={self.input_transform}, ' \
#             f'ignore_index={self.ignore_index}, ' \
#             f'align_corners={self.align_corners}'
#         return s

#     def _init_inputs(self, in_channels, in_index, input_transform):
#         """Check and initialize input transforms.

#         The in_channels, in_index and input_transform must match.
#         Specifically, when input_transform is None, only single feature map
#         will be selected. So in_channels and in_index must be of type int.
#         When input_transform

#         Args:
#             in_channels (int|Sequence[int]): Input channels.
#             in_index (int|Sequence[int]): Input feature index.
#             input_transform (str|None): Transformation type of input features.
#                 Options: 'resize_concat', 'multiple_select', None.
#                 'resize_concat': Multiple feature maps will be resize to the
#                     same size as first one and than concat together.
#                     Usually used in FCN head of HRNet.
#                 'multiple_select': Multiple feature maps will be bundle into
#                     a list and passed into decode head.
#                 None: Only one select feature map is allowed.
#         """

#         if input_transform is not None:
#             assert input_transform in ['resize_concat', 'multiple_select']
#         self.input_transform = input_transform
#         self.in_index = in_index
#         if input_transform is not None:
#             assert isinstance(in_channels, (list, tuple))
#             assert isinstance(in_index, (list, tuple))
#             assert len(in_channels) == len(in_index)
#             if input_transform == 'resize_concat':
#                 self.in_channels = sum(in_channels)
#             else:
#                 self.in_channels = in_channels
#         else:
#             assert isinstance(in_channels, int)
#             assert isinstance(in_index, int)
#             self.in_channels = in_channels

#     def init_weights(self):
#         """Initialize weights of classification layer."""
#         # normal_init(self.conv_seg, mean=0, std=0.01)
#         # normal_init(self.conv1, mean=0, std=0.01)

#     def _transform_inputs(self, inputs):
#         """Transform inputs for decoder.

#         Args:
#             inputs (list[Tensor]): List of multi-level img features.

#         Returns:
#             Tensor: The transformed inputs
#         """

#         if self.input_transform == 'resize_concat':
#             inputs = [inputs[i] for i in self.in_index]
#             upsampled_inputs = [
#                 resize(
#                     input=x,
#                     size=inputs[0].shape[2:],
#                     mode='bilinear',
#                     align_corners=self.align_corners) for x in inputs
#             ]
#             inputs = torch.cat(upsampled_inputs, dim=1)
#         elif self.input_transform == 'multiple_select':
#             inputs = [inputs[i] for i in self.in_index]
#         else:
#             inputs = inputs[self.in_index]

#         return inputs

#     def forward(self, inputs):
#         """Placeholder of forward function."""
#         pass

#     def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
#         """Forward function for training.
#         Args:
#             inputs (list[Tensor]): List of multi-level img features.
#             img_metas (list[dict]): List of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmseg/datasets/pipelines/formatting.py:Collect`.
#             gt_semantic_seg (Tensor): Semantic segmentation masks
#                 used if the architecture supports semantic segmentation task.
#             train_cfg (dict): The training config.

#         Returns:
#             dict[str, Tensor]: a dictionary of loss components
#         """
#         seg_logits = self.forward(inputs)
#         losses = self.losses(seg_logits, gt_semantic_seg)
#         return losses

#     def forward_test(self, inputs, img_metas, test_cfg):
#         """Forward function for testing.

#         Args:
#             inputs (list[Tensor]): List of multi-level img features.
#             img_metas (list[dict]): List of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmseg/datasets/pipelines/formatting.py:Collect`.
#             test_cfg (dict): The testing config.

#         Returns:
#             Tensor: Output segmentation map.
#         """
#         return self.forward(inputs)


# class UPerHead(BaseDecodeHead):
#     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
#         super(UPerHead, self).__init__(
#             input_transform='multiple_select', **kwargs)
#         # FPN Module
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()
#         for in_channels in self.in_channels:  # skip the top layer
#             l_conv = ConvModule(
#                 in_channels,
#                 self.channels,
#                 1,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg,
#                 inplace=True)
#             fpn_conv = ConvModule(
#                 self.channels,
#                 self.channels,
#                 3,
#                 padding=1,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg,
#                 inplace=True)
#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)

#     def forward(self, inputs):
#         """Forward function."""

#         inputs = self._transform_inputs(inputs)

#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]

#         # laterals.append(self.psp_forward(inputs))

#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             prev_shape = laterals[i - 1].shape[2:]
#             laterals[i - 1] += resize(
#                 laterals[i],
#                 size=prev_shape,
#                 mode='bilinear',
#                 align_corners=self.align_corners)

#         # build outputs
#         fpn_outs = [
#             self.fpn_convs[i](laterals[i])
#             for i in range(used_backbone_levels - 1)
#         ]
#         # append psp feature
#         fpn_outs.append(laterals[-1])

#         return fpn_outs[0]



# class PSP(BaseDecodeHead):
#     """Unified Perceptual Parsing for Scene Understanding.

#     This head is the implementation of `UPerNet
#     <https://arxiv.org/abs/1807.10221>`_.

#     Args:
#         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
#             Module applied on the last feature. Default: (1, 2, 3, 6).
#     """

#     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
#         super(PSP, self).__init__(
#             input_transform='multiple_select', **kwargs)
#         # PSP Module
#         self.psp_modules = PPM(
#             pool_scales,
#             self.in_channels[-1],
#             self.channels,
#             # conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg,
#             align_corners=self.align_corners)
#         self.bottleneck = ConvModule(
#             self.in_channels[-1] + len(pool_scales) * self.channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)

#     def psp_forward(self, inputs):
#         """Forward function of PSP module."""
#         x = inputs[-1]
#         psp_outs = [x]
#         psp_outs.extend(self.psp_modules(x))
#         psp_outs = torch.cat(psp_outs, dim=1)
#         output = self.bottleneck(psp_outs)

#         return output

#     def forward(self, inputs):
#         """Forward function."""
#         inputs = self._transform_inputs(inputs)
        
#         return self.psp_forward(inputs)


# # swin
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# # from .newcrf_utils import load_checkpoint


# class swin_Mlp(nn.Module):
#     """ Multilayer perceptron."""

#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# def swin_window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows


# def swin_window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x


# class swin_WindowAttention(nn.Module):
#     """ Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """ Forward function.

#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class SwinTransformerBlock(nn.Module):
#     """ Swin Transformer Block.

#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, dim, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
#         self.attn = swin_WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = swin_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.H = None
#         self.W = None

#     def forward(self, x, mask_matrix):
#         """ Forward function.

#         Args:
#             x: Input feature, tensor size (B, H*W, C).
#             H, W: Spatial resolution of the input feature.
#             mask_matrix: Attention mask for cyclic shift.
#         """
#         B, L, C = x.shape
#         H, W = self.H, self.W
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)

#         # pad feature maps to multiples of window size
#         pad_l = pad_t = 0
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
#         _, Hp, Wp, _ = x.shape

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#             attn_mask = mask_matrix
#         else:
#             shifted_x = x
#             attn_mask = None

#         # partition windows
#         x_windows = swin_window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = swin_window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x

#         if pad_r > 0 or pad_b > 0:
#             x = x[:, :H, :W, :].contiguous()

#         x = x.view(B, H * W, C)

#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x


# class PatchMerging(nn.Module):
#     """ Patch Merging Layer

#     Args:
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)

#     def forward(self, x, H, W):
#         """ Forward function.

#         Args:
#             x: Input feature, tensor size (B, H*W, C).
#             H, W: Spatial resolution of the input feature.
#         """
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         x = x.view(B, H, W, C)

#         # padding
#         pad_input = (H % 2 == 1) or (W % 2 == 1)
#         if pad_input:
#             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x)

#         return x


# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.

#     Args:
#         dim (int): Number of feature channels
#         depth (int): Depths of this stage.
#         num_heads (int): Number of attention head.
#         window_size (int): Local window size. Default: 7.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(self,
#                  dim,
#                  depth,
#                  num_heads,
#                  window_size=7,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  norm_layer=nn.LayerNorm,
#                  downsample=None,
#                  use_checkpoint=False):
#         super().__init__()
#         self.window_size = window_size
#         self.shift_size = window_size // 2
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint

#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(
#                 dim=dim,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else window_size // 2,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer)
#             for i in range(depth)])

#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None

#     def forward(self, x, H, W):
#         """ Forward function.

#         Args:
#             x: Input feature, tensor size (B, H*W, C).
#             H, W: Spatial resolution of the input feature.
#         """

#         # calculate attention mask for SW-MSA
#         Hp = int(np.ceil(H / self.window_size)) * self.window_size
#         Wp = int(np.ceil(W / self.window_size)) * self.window_size
#         img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1

#         mask_windows = swin_window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

#         for blk in self.blocks:
#             blk.H, blk.W = H, W
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, attn_mask)
#             else:
#                 x = blk(x, attn_mask)
#         if self.downsample is not None:
#             x_down = self.downsample(x, H, W)
#             Wh, Ww = (H + 1) // 2, (W + 1) // 2
#             return x, H, W, x_down, Wh, Ww
#         else:
#             return x, H, W, x, H, W


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding

#     Args:
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         patch_size = to_2tuple(patch_size)
#         self.patch_size = patch_size

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         """Forward function."""
#         # padding
#         _, _, H, W = x.size()
#         if W % self.patch_size[1] != 0:
#             x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
#         if H % self.patch_size[0] != 0:
#             x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

#         x = self.proj(x)  # B C Wh Ww
#         if self.norm is not None:
#             Wh, Ww = x.size(2), x.size(3)
#             x = x.flatten(2).transpose(1, 2)
#             x = self.norm(x)
#             x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

#         return x


# class SwinTransformer(nn.Module):
#     """ Swin Transformer backbone.
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030

#     Args:
#         pretrain_img_size (int): Input image size for training the pretrained model,
#             used in absolute postion embedding. Default 224.
#         patch_size (int | tuple(int)): Patch size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         depths (tuple[int]): Depths of each Swin Transformer stage.
#         num_heads (tuple[int]): Number of attention head of each stage.
#         window_size (int): Window size. Default: 7.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
#         drop_rate (float): Dropout rate.
#         attn_drop_rate (float): Attention dropout rate. Default: 0.
#         drop_path_rate (float): Stochastic depth rate. Default: 0.2.
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True.
#         out_indices (Sequence[int]): Output from which stages.
#         frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
#             -1 means not freezing any parameters.
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(self,
#                  pretrain_img_size=224,
#                  patch_size=4,
#                  in_chans=3,
#                  embed_dim=96,
#                  depths=[2, 2, 6, 2],
#                  num_heads=[3, 6, 12, 24],
#                  window_size=7,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.2,
#                  norm_layer=nn.LayerNorm,
#                  ape=False,
#                  patch_norm=True,
#                  out_indices=(0, 1, 2, 3),
#                  frozen_stages=-1,
#                  use_checkpoint=False):
#         super().__init__()

#         self.pretrain_img_size = pretrain_img_size
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.out_indices = out_indices
#         self.frozen_stages = frozen_stages

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)

#         # absolute position embedding
#         if self.ape:
#             pretrain_img_size = to_2tuple(pretrain_img_size)
#             patch_size = to_2tuple(patch_size)
#             patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
#             trunc_normal_(self.absolute_pos_embed, std=.02)

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(
#                 dim=int(embed_dim * 2 ** i_layer),
#                 depth=depths[i_layer],
#                 num_heads=num_heads[i_layer],
#                 window_size=window_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                 norm_layer=norm_layer,
#                 downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                 use_checkpoint=use_checkpoint)
#             self.layers.append(layer)

#         num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
#         self.num_features = num_features

#         # add a norm layer for each output
#         for i_layer in out_indices:
#             layer = norm_layer(num_features[i_layer])
#             layer_name = f'norm{i_layer}'
#             self.add_module(layer_name, layer)

#         self._freeze_stages()

#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             self.patch_embed.eval()
#             for param in self.patch_embed.parameters():
#                 param.requires_grad = False

#         if self.frozen_stages >= 1 and self.ape:
#             self.absolute_pos_embed.requires_grad = False

#         if self.frozen_stages >= 2:
#             self.pos_drop.eval()
#             for i in range(0, self.frozen_stages - 1):
#                 m = self.layers[i]
#                 m.eval()
#                 for param in m.parameters():
#                     param.requires_grad = False

#     def init_weights(self, pretrained=None):
#         """Initialize the weights in backbone.

#         Args:
#             pretrained (str, optional): Path to pre-trained weights.
#                 Defaults to None.
#         """

#         def _init_weights(m):
#             if isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=.02)
#                 if isinstance(m, nn.Linear) and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#         if isinstance(pretrained, str):
#             self.apply(_init_weights)
#             # logger = get_root_logger()
#             load_checkpoint(self, pretrained, strict=False)
#         elif pretrained is None:
#             self.apply(_init_weights)
#         else:
#             raise TypeError('pretrained must be a str or None')

#     def forward(self, x):
#         """Forward function."""
#         x = self.patch_embed(x)

#         Wh, Ww = x.size(2), x.size(3)
#         if self.ape:
#             # interpolate the position embedding to the corresponding size
#             absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
#             x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
#         else:
#             x = x.flatten(2).transpose(1, 2)
#         x = self.pos_drop(x)

#         outs = []
#         for i in range(self.num_layers):
#             layer = self.layers[i]
#             x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

#             if i in self.out_indices:
#                 norm_layer = getattr(self, f'norm{i}')
#                 x_out = norm_layer(x_out)

#                 out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
#                 outs.append(out)

#         return tuple(outs)

#     def train(self, mode=True):
#         """Convert the model into training mode while keep layers freezed."""
#         super(SwinTransformer, self).train(mode)
#         self._freeze_stages()




# # SOTA depth
# def convnext_large(pretrained=True, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     return model

# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
#     with shape (batch_size, channels, height, width).
#     """
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError 
#         self.normalized_shape = (normalized_shape, )
    
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x

# class ConvNeXt(nn.Module):
#     r""" ConvNeXt
#         A PyTorch impl of : `A ConvNet for the 2020s`  -
#           https://arxiv.org/pdf/2201.03545.pdf
#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """
#     def __init__(self, in_chans=3, num_classes=1000, 
#                  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
#                  layer_scale_init_value=1e-6, head_init_scale=1.,
#                  **kwargs,):
#         super().__init__()

#         self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
#         stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )
#         self.downsample_layers.append(stem)
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                     LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                     nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
#             )
#             self.downsample_layers.append(downsample_layer)

#         self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
#         dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
#                 layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             nn.init.constant_(m.bias, 0)

#     def forward_features(self, x):
#         features = []
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             x = self.stages[i](x)
#             features.append(x)
#         return features # global average pooling, (N, C, H, W) -> (N, C)

#     def forward(self, x):
#         features = self.forward_features(x)
#         return features

# class ConvNeXtBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
    
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

#         x = input + self.drop_path(x)
#         return x

# def compute_depth_expectation(prob, depth_values):
#     depth_values = depth_values.view(*depth_values.shape, 1, 1)
#     depth = torch.sum(prob * depth_values, 1)
#     return depth

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(ConvBlock, self).__init__()

#         if kernel_size == 3:
#             self.conv = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1),
#             )
#         elif kernel_size == 1:
#             self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, padding=0, stride=1)

#         self.nonlin = nn.ELU(inplace=True)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.nonlin(out)
#         return out


# class ConvBlock_double(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(ConvBlock_double, self).__init__()

#         if kernel_size == 3:
#             self.conv = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1),
#             )
#         elif kernel_size == 1:
#             self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, padding=0, stride=1)

#         self.nonlin = nn.ELU(inplace=True)
#         self.conv_2 = nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1)
#         self.nonlin_2 =nn.ELU(inplace=True)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.nonlin(out)
#         out = self.conv_2(out)
#         out = self.nonlin_2(out)
#         return out
    
# class DecoderFeature(nn.Module):
#     def __init__(self, feat_channels, num_ch_dec=[64, 64, 128, 256]):
#         super(DecoderFeature, self).__init__()
#         self.num_ch_dec = num_ch_dec
#         self.feat_channels = feat_channels

#         self.upconv_3_0 = ConvBlock(self.feat_channels[3], self.num_ch_dec[3], kernel_size=1)
#         self.upconv_3_1 = ConvBlock_double(
#             self.feat_channels[2] + self.num_ch_dec[3],
#             self.num_ch_dec[3],
#             kernel_size=1)
        
#         self.upconv_2_0 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2], kernel_size=3)
#         self.upconv_2_1 = ConvBlock_double(
#             self.feat_channels[1] + self.num_ch_dec[2],
#             self.num_ch_dec[2],
#             kernel_size=3)
        
#         self.upconv_1_0 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1], kernel_size=3)
#         self.upconv_1_1 = ConvBlock_double(
#             self.feat_channels[0] + self.num_ch_dec[1],
#             self.num_ch_dec[1],
#             kernel_size=3)
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#     def forward(self, ref_feature):
#         x = ref_feature[3]

#         x = self.upconv_3_0(x)
#         x = torch.cat((self.upsample(x), ref_feature[2]), 1)
#         x = self.upconv_3_1(x)

#         x = self.upconv_2_0(x)
#         x = torch.cat((self.upsample(x), ref_feature[1]), 1)
#         x = self.upconv_2_1(x)

#         x = self.upconv_1_0(x)
#         x = torch.cat((self.upsample(x), ref_feature[0]), 1)
#         x = self.upconv_1_1(x)
#         return x


# class UNet(nn.Module):
#     def __init__(self, inp_ch=32, output_chal=1, down_sample_times=3, channel_mode='v0'):
#         super(UNet, self).__init__()
#         basic_block = ConvBnReLU
#         num_depth = 128

#         self.conv0 = basic_block(inp_ch, num_depth)
#         if channel_mode == 'v0':
#             channels = [num_depth, num_depth//2, num_depth//4, num_depth//8, num_depth // 8]
#         elif channel_mode == 'v1':
#             channels = [num_depth, num_depth, num_depth, num_depth, num_depth, num_depth]
#         self.down_sample_times = down_sample_times
#         for i in range(down_sample_times):
#             setattr(
#                 self, 'conv_%d' % i,
#                 nn.Sequential(
#                     basic_block(channels[i], channels[i+1], stride=2),
#                     basic_block(channels[i+1], channels[i+1])
#                 )
#             )
#         for i in range(down_sample_times-1,-1,-1):
#             setattr(self, 'deconv_%d' % i,
#                     nn.Sequential(
#                         nn.ConvTranspose2d(
#                             channels[i+1],
#                             channels[i],
#                             kernel_size=3,
#                             padding=1,
#                             output_padding=1,
#                             stride=2,
#                             bias=False),
#                         nn.BatchNorm2d(channels[i]),
#                         nn.ReLU(inplace=True)
#                     )
#                 )
#             self.prob = nn.Conv2d(num_depth, output_chal, 1, stride=1, padding=0)
    
#     def forward(self, x):
#         features = {}
#         conv0 = self.conv0(x)
#         x = conv0
#         features[0] = conv0
#         for i in range(self.down_sample_times):
#             x = getattr(self, 'conv_%d' % i)(x)
#             features[i+1] = x
#         for i in range(self.down_sample_times-1,-1,-1):
#             x = features[i] + getattr(self, 'deconv_%d' % i)(x)
#         x = self.prob(x)
#         return x

# class ConvBnReLU(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
#         super(ConvBnReLU, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=pad,
#             bias=False
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
    
#     def forward(self, x):
#         return F.relu(self.bn(self.conv(x)), inplace=True)


# class HourglassDecoder(nn.Module):
#     def __init__(self):
#         super(HourglassDecoder, self).__init__()
#         self.inchannels = [192, 384, 768, 1536]#cfg.model.decode_head.in_channels #  [256, 512, 1024, 2048]
#         self.decoder_channels = [128, 128, 256, 512]#cfg.model.decode_head.decoder_channel # [64, 64, 128, 256]
#         self.min_val = 0.3#cfg.data_basic.depth_normalize[0]
#         self.max_val = 150#cfg.data_basic.depth_normalize[1]

#         self.num_ch_dec = self.decoder_channels # [64, 64, 128, 256]
#         self.num_depth_regressor_anchor = 512
#         self.feat_channels = self.inchannels
#         unet_in_channel = self.num_ch_dec[1]
#         unet_out_channel = 256

#         self.decoder_mono = DecoderFeature(self.feat_channels, self.num_ch_dec)
#         self.conv_out_2 = UNet(inp_ch=unet_in_channel,
#                                output_chal=unet_out_channel + 1,
#                                down_sample_times=3,
#                                channel_mode='v0',
#                                )

#         self.depth_regressor_2 = nn.Sequential(
#             nn.Conv2d(unet_out_channel,
#                       self.num_depth_regressor_anchor,
#                       kernel_size=3,
#                       padding=1,
#                 ),
#             nn.BatchNorm2d(self.num_depth_regressor_anchor),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 self.num_depth_regressor_anchor,
#                 self.num_depth_regressor_anchor,
#                 kernel_size=1,
#             )
#         )
#         self.residual_channel = 16
#         self.conv_up_2 = nn.Sequential(
#             nn.Conv2d(1 + 2 + unet_out_channel, self.residual_channel, 3, padding=1),
#             nn.BatchNorm2d(self.residual_channel),
#             nn.ReLU(),
#             nn.Conv2d(self.residual_channel, self.residual_channel, 3, padding=1),
#             nn.Upsample(scale_factor=4),
#             nn.Conv2d(self.residual_channel, self.residual_channel, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(self.residual_channel, 1, 1, padding=0),
#         )
    
#     def get_bins(self, bins_num):
#         depth_bins_vec = torch.linspace(math.log(self.min_val), math.log(self.max_val), bins_num, device='cuda')
#         depth_bins_vec = torch.exp(depth_bins_vec)
#         return depth_bins_vec
    
#     def register_depth_expectation_anchor(self, bins_num, B):
#         depth_bins_vec = self.get_bins(bins_num)
#         depth_bins_vec = depth_bins_vec.unsqueeze(0).repeat(B, 1)
#         self.register_buffer('depth_expectation_anchor', depth_bins_vec, persistent=False)

#     def upsample(self, x, scale_factor=2):
#         return F.interpolate(x, scale_factor=scale_factor, mode='nearest')

#     def regress_depth_2(self, feature_map_d):
#         prob = self.depth_regressor_2(feature_map_d).softmax(dim=1)
#         B = prob.shape[0]
#         if "depth_expectation_anchor" not in self._buffers:
#             self.register_depth_expectation_anchor(self.num_depth_regressor_anchor, B)
#         d = compute_depth_expectation(
#             prob,
#             self.depth_expectation_anchor[:B, ...]
#         ).unsqueeze(1)
#         return d

#     def create_mesh_grid(self, height, width, batch, device="cuda", set_buffer=True):
#         y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
#                                torch.arange(0, width, dtype=torch.float32, device=device)], indexing='ij')
#         meshgrid = torch.stack((x, y))
#         meshgrid = meshgrid.unsqueeze(0).repeat(batch, 1, 1, 1)
#         return meshgrid

#     def forward(self, features_mono, **kwargs):
#         '''
#         trans_ref2src: list of transformation matrix from the reference view to source view. [B, 4, 4]
#         inv_intrinsic_pool: list of inverse intrinsic matrix.
#         features_mono: features of reference and source views. [[ref_f1, ref_f2, ref_f3, ref_f4],[src1_f1, src1_f2, src1_f3, src1_f4], ...].
#         '''
#         outputs = {}
#         # get encoder feature of the reference view
#         ref_feat = features_mono

#         feature_map_mono = self.decoder_mono(ref_feat)
#         feature_map_mono_pred = self.conv_out_2(feature_map_mono)
#         confidence_map_2 = feature_map_mono_pred[:, -1:, :, :]
#         feature_map_d_2 = feature_map_mono_pred[:, :-1, :, :]

#         depth_pred_2 = self.regress_depth_2(feature_map_d_2)

#         B, _, H, W = depth_pred_2.shape

#         meshgrid = self.create_mesh_grid(H, W, B)

#         depth_pred_mono = self.upsample(depth_pred_2, scale_factor=4) + 1e-1 * \
#             self.conv_up_2(
#                 torch.cat((depth_pred_2, meshgrid[:B, ...], feature_map_d_2), 1)
#             )
#         confidence_map_mono = self.upsample(confidence_map_2, scale_factor=4)

#         outputs=dict(
#             prediction=depth_pred_mono,
#             confidence=confidence_map_mono,
#             pred_logit=None,
#         )
#         return outputs
    
    



