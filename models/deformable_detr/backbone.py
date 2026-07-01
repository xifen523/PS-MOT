# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified for PS-MOT (ECCV 2026), CVPU Kai Luo. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import ResNet50_Weights
from typing import Dict, List

from utils.nested_tensor import NestedTensor
from utils.misc import is_main_process

from models.deformable_detr.position_encoding import build_position_encoding
class PEWAModule(nn.Module):
    """
    PEWA.
    """
    def __init__(self, channels):
        super(PEWAModule, self).__init__()
        self.channels = channels
        
        self.hf_modulator = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )
        self.lf_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def dwt_haar(self, x):
        pad_h = x.shape[2] % 2
        pad_w = x.shape[3] % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        
        x_LL = (x00 + x01 + x10 + x11) / 4.0
        x_HL = (x00 - x01 + x10 - x11) / 4.0
        x_LH = (x00 + x01 - x10 - x11) / 4.0
        x_HH = (x00 - x01 - x10 + x11) / 4.0
        
        return x_LL, torch.cat([x_HL, x_LH, x_HH], dim=1)

    def idwt_haar(self, x_LL, x_HF):
        C = x_LL.size(1)
        x_HL, x_LH, x_HH = x_HF[:, :C], x_HF[:, C:2*C], x_HF[:, 2*C:]

        h = torch.zeros(
            x_LL.size(0), C, x_LL.size(2) * 2, x_LL.size(3) * 2,
            device=x_LL.device, dtype=x_LL.dtype
        )
        h[:, :, 0::2, 0::2] = x_LL + x_HL + x_LH + x_HH
        h[:, :, 0::2, 1::2] = x_LL - x_HL + x_LH - x_HH
        h[:, :, 1::2, 0::2] = x_LL + x_HL - x_LH - x_HH
        h[:, :, 1::2, 1::2] = x_LL - x_HL - x_LH + x_HH
        return h

    def forward(self, x, heatmap):
        identity = x
        orig_h, orig_w = x.shape[-2:]
        pad_h = orig_h % 2
        pad_w = orig_w % 2

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_LL, x_HF = self.dwt_haar(x)

        h_low = F.interpolate(heatmap, size=x_LL.shape[-2:], mode='bilinear', align_corners=False)
        hf_weights = self.hf_modulator(h_low).repeat_interleave(self.channels, dim=1)
        x_HF_excited = x_HF * hf_weights

        x_LL_calibrated = x_LL * self.lf_calibration(x_LL)

        out = self.idwt_haar(x_LL_calibrated, x_HF_excited)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :orig_h, :orig_w]

        return out + identity

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# class BackboneBase(nn.Module):

#     def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
#         super().__init__()
#         for name, parameter in backbone.named_parameters():
#             if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#                 parameter.requires_grad_(False)
#         if return_interm_layers:
#             # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
#             return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
#             self.strides = [8, 16, 32]
#             self.num_channels = [512, 1024, 2048]
#         else:
#             return_layers = {'layer4': "0"}
#             self.strides = [32]
#             self.num_channels = [2048]
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

#     def forward(self, tensor_list: NestedTensor):
#         xs = self.body(tensor_list.tensors)
#         out: Dict[str, NestedTensor] = {}
#         for name, x in xs.items():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out[name] = NestedTensor(x, mask)
#         return out

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        self.pewa_modules = nn.ModuleDict({
            name: PEWAModule(ch) for name, ch in zip(return_layers.values(), self.num_channels)
        })

    def generate_heatmap(self, points_list, shape):
        B, _, H, W = shape
        device = points_list[0].device if len(points_list) > 0 else torch.device("cpu")
        heatmap = torch.zeros((B, 1, H, W), device=device)
        sigma = 3.0
        for b, pts in enumerate(points_list):
            if pts.numel() == 0:
                continue
            coords = pts.clone()
            coords[:, 0] *= W
            coords[:, 1] *= H
            for pt in coords:
                y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                dist = (x - pt[0])**2 + (y - pt[1])**2
                heatmap[b, 0] += torch.exp(-dist / (2 * sigma**2))
        return torch.clamp(heatmap, 0, 1)

    def forward(self, tensor_list: NestedTensor, points=None):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            if points is not None:
                h_map = self.generate_heatmap(points, x.shape)
                x = self.pewa_modules[name](x, h_map)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        if name == "resnet50":
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=ResNet50_Weights.IMAGENET1K_V1 if is_main_process() else None, norm_layer=norm_layer)
        else:
            raise NotImplementedError(f"Do not support backbone name {name}.")
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding):
#         super().__init__(backbone, position_embedding)
#         self.strides = backbone.strides
#         self.num_channels = backbone.num_channels

#     def forward(self, tensor_list: NestedTensor):
#         xs = self[0](tensor_list)
#         out: List[NestedTensor] = []
#         pos = []
#         for name, x in sorted(xs.items()):
#             out.append(x)

#         # position encoding
#         for x in out:
#             pos.append(self[1](x).to(x.tensors.dtype))

#         return out, pos
    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor, points=None):
        xs = self[0](tensor_list, points=points)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_intern_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_intern_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
