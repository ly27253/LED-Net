import torch
import torch.nn as nn
import time
from model_utils_speed import BasicBlock, Bottleneck, segmenthead, AFF, ASPP, CARAFE, segmentheadCARAFE, iAFF, segmenthead_drop, Muti_AFF, segmenthead_c, SPASPP, MFACB
from thop import profile
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule, Sequential

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from mmseg.models.backbones.stdc import STDCModule
from mmseg.models.backbones.UNetFormer_GETB import GlobalLocalAttention, Block

# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType


@MODELS.register_module()
class DDRNet_yuanban(BaseModule):
    """DDRNet backbone.

    This backbone is the implementation of `Deep Dual-resolution Networks for
    Real-time and Accurate Semantic Segmentation of Road Scenes
    <http://arxiv.org/abs/2101.06085>`_.
    Modified from https://github.com/ydhongHIT/DDRNet.

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        channels: (int): The base channels of DDRNet. Default: 32.
        ppm_channels (int): The channels of PPM module. Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        # low resolution(context) branch
        self.context_branch_layers = nn.ModuleList()
        for i in range(3):
            self.context_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2**(i + 1),
                    planes=channels * 8 if i > 0 else channels * 4,
                    num_blocks=2 if i < 2 else 1,
                    stride=2))

        # bilateral fusion
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))

        # high resolution(spatial) branch
        self.spatial_branch_layers = nn.ModuleList()
        for i in range(3):
            self.spatial_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2,
                    planes=channels * 2,
                    num_blocks=2 if i < 2 else 1,
                ))

        self.spp = DAPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5)

    def _make_stem_layer(self, in_channels, channels, num_blocks):
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.extend([
            self._make_layer(BasicBlock, channels, channels, num_blocks),
            nn.ReLU(),
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2),
            nn.ReLU(),
        ])

        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        # out_size = (x.shape[-2] // 8, x.shape[-1] // 8)
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))  # 0527xiugai 向上取zheng

        # stage 0-2
        x = self.stem(x)   # 1/8

        # stage3
        x_c = self.context_branch_layers[0](x)
        x_s = self.spatial_branch_layers[0](x)
        comp_c = self.compression_1(self.relu(x_c))
        x_c += self.down_1(self.relu(x_s))
        x_s += resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_context = x_s.clone()   # 1/8

        # stage4
        x_c = self.context_branch_layers[1](self.relu(x_c))
        x_s = self.spatial_branch_layers[1](self.relu(x_s))
        comp_c = self.compression_2(self.relu(x_c))
        x_c += self.down_2(self.relu(x_s))
        x_s += resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        # stage5
        x_s = self.spatial_branch_layers[2](self.relu(x_s))   # RBB 1/8
        x_c = self.context_branch_layers[2](self.relu(x_c))
        x_c = self.spp(x_c)  # DAPPM
        x_c = resize(
            x_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        return (temp_context, x_s + x_c) if self.training else x_s + x_c



def get_pred_model(name, num_classes):
    if 's' in name:
        model = DDRNet_yuanban()

    return model


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    model = get_pred_model(name='ds_s', num_classes=19)
    model.eval()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    iterations = None
    # print(model)
    input = torch.randn(1, 3, 1024, 2048).cuda()
    # input = torch.randn(1, 3, 512, 1024).cuda()
    # print(model)
    flops, params = profile(model.to(device), inputs=(input,))

    print("参数量：", params)
    print("FLOPS：", flops/(1024*1024*1024))

    # 在前向传播之前
    start_memory = torch.cuda.memory_allocated()

    # 前向传播
    output = model(input)

    # 在前向传播之后
    end_memory = torch.cuda.memory_allocated()
    memory_used = end_memory - start_memory

    print(f"GPU Memory Used: {memory_used / 1024 ** 2} MB")


    tt = []


    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()

        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_end = time.time()
        elapsed_time = t_end- t_start
        ms = elapsed_time/iterations
        latency = elapsed_time / iterations * 1000

        print("迭代轮次：", iterations)

    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print("FPS:", FPS)
    print("每张图像推理时间：", ms)




