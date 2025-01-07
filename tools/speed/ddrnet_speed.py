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

import numpy as np

def normalize_tensor(mytensor):
    """
    将输入的张量归一化到 [0, 1] 范围。

    参数:
    tensor (torch.Tensor): 输入的张量。

    返回:
    torch.Tensor: 归一化后的张量。
    """
    min_val = torch.min(mytensor)
    max_val = torch.max(mytensor)
    normalized_tensor = (mytensor - min_val) / (max_val - min_val)
    return normalized_tensor

@MODELS.register_module()
class DDRNet1(BaseModule):
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
                 num_convs=4,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 planes=64,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        stride_2 = 2
        stride_1 = 1
        self.stride_2 = stride_2
        self.stride_1 = stride_1
        self.num_convs = num_convs
        self.channels = (64, 128, 256, 512, 1024)
        self.channels_spatial = (64, 64, 64, 128, 256)
        self.gltb1 = Block(dim=128, num_heads=8, window_size=8)
        self.gltb2 = Block(dim=256, num_heads=8, window_size=8)
        self.gltb3 = Block(dim=128, num_heads=8, window_size=8)

        self.aff1 = Muti_AFF(channels=channels * 2)
        self.aff2 = Muti_AFF(channels=channels * 2)

        self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],dtype=torch.float32,requires_grad=False).reshape((1, 1, 3, 3)).cuda()
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
            dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)
        self.boundary_threshold = 0.1

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        self.conv_1 = ConvModule(
            64,
            1,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=None, padding=1)
        self.conv_2 = ConvModule(
            1,
            64,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=None, padding=1)

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        # low resolution(context) branch
        self.context_branch_layers = nn.ModuleList()
        for i in range(3):
            self.context_branch_layers.append(self._make_stdc_stage(self.channels[i], self.channels[i+1],
                                                             self.stride_2, norm_cfg, act_cfg, bottleneck_type='cat'))

        # bilateral fusion
        # self.compression_1 = ConvModule(
        #     channels * 4,
        #     channels * 2,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)

        self.compression_aff = ConvModule(
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
            self.spatial_branch_layers.append(self._make_stdc_stage(self.channels_spatial[i], self.channels_spatial[i + 1],
                                            self.stride_1, norm_cfg, act_cfg, bottleneck_type='cat'))

        # self.layer3 = nn.Sequential(
        #         MFACB(planes * 1, planes * 1, planes * 2, stride_1=2, dilation=[2, 2, 2]),
        #         MFACB(planes * 2, planes * 2, planes * 2, dilation=[2, 2, 2]),
        #         MFACB(planes * 2, planes * 2, planes * 2, dilation=[3, 3, 3]),
        # )
        #
        # self.layer4 = nn.Sequential(
        #         MFACB(planes * 2, planes * 2, planes * 4, stride_1=2, dilation=[3, 3, 3]),
        #         # MFACB(planes * 4, planes * 4, planes * 4, dilation=[5, 5, 5]),
        # )
        #
        # self.layer5 = nn.Sequential(
        #         MFACB(planes * 4, planes * 4, planes * 8, stride_1=2, dilation=[3, 3, 3]),
        #         # MFACB(planes * 8, planes * 8, planes * 8, dilation=[5, 5, 5]),
        # )

        self.spp = DAPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5)
        # self.spp_end = SPASPP(128, 128, 128)

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

    def _make_stdc_stage(self, in_channels, out_channels, stride, norm_cfg,
                    act_cfg, bottleneck_type= 'cat'):
        layers = []
        layers.append(
            STDCModule(
                in_channels,
                out_channels,
                stride,
                norm_cfg,
                act_cfg,
                num_convs=self.num_convs,
                fusion_type=bottleneck_type))
        return Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        # out_size = (x.shape[-2] // 8, x.shape[-1] // 8)
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))  # 0527xiugai 向上取zheng

        # stage 0-2
        x = self.stem(x)   # 1/8

        seg_label = self.conv_1(x)
        seg_label = normalize_tensor(seg_label)
        # seg_label1 = np.array(seg_label.detach().cpu())

        boundary_targets = F.conv2d(
            seg_label, self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        # 将张量归一化到 [0, 1] 范围
        # boundary_targets = normalize_tensor(boundary_targets)
        # boundary_targets1 = np.array(boundary_targets.detach().cpu())
        boundary_targets[boundary_targets > self.boundary_threshold] = 1
        boundary_targets[boundary_targets <= self.boundary_threshold] = 0
        # boundary_targets1 = np.array(boundary_targets.detach().cpu())

        boundary_targets_x2 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x4_up = F.interpolate(
            boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(
            boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        # 将张量归一化到 [0, 1] 范围
        # boundary_targets_x2_up = normalize_tensor(boundary_targets_x2_up)
        # boundary_targets_x2_up1 = np.array(boundary_targets_x2_up.detach().cpu())
        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0

        # 将张量归一化到 [0, 1] 范围
        # boundary_targets_x4_up = normalize_tensor(boundary_targets_x4_up)
        # boundary_targets_x4_up1 = np.array(boundary_targets_x4_up.detach().cpu())
        boundary_targets_x4_up[
            boundary_targets_x4_up > self.boundary_threshold] = 1
        boundary_targets_x4_up[
            boundary_targets_x4_up <= self.boundary_threshold] = 0

        boundary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
            dim=1)

        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boundary_targets_pyramids,
                                           self.fusion_kernel)

        boudary_targets_pyramid[
            boudary_targets_pyramid > self.boundary_threshold] = 1
        boudary_targets_pyramid[
            boudary_targets_pyramid <= self.boundary_threshold] = 0

        seg_labels = boudary_targets_pyramid.float()

        # stage3
        x_c = self.context_branch_layers[0](x)
        # x_c = self.layer3[0](x)
        # UNETFormer GLTB   128*32*64
        x_c = self.gltb1(x_c)
        x_s = self.spatial_branch_layers[0](x)
        comp_c = self.compression_aff(self.relu(x_c))
        x_c = x_c + self.down_1(self.relu(x_s))
        # x_s += resize(
        #     comp_c,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        comp_c = resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        # MSAF
        x_s = self.aff1(x_s, comp_c)

        if self.training:
            temp_context = x_s.clone()   # 1/8

        # stage4
        x_c = self.context_branch_layers[1](self.relu(x_c))
        # x_c = self.layer4(self.relu(x_c))
        # UNETFormer GLTB   256*16*32
        x_c = self.gltb2(x_c)
        x_s = self.spatial_branch_layers[1](self.relu(x_s))
        comp_c = self.compression_2(self.relu(x_c))
        x_c = x_c + self.down_2(self.relu(x_s))
        # x_s += resize(
        #     comp_c,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        comp_c = resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        # MSAF
        x_s = self.aff2(x_s, comp_c)
        result = self.conv_2(seg_labels) * x_s
        x_s = result + x_s

        # stage5
        x_s = self.spatial_branch_layers[2](self.relu(x_s))   # RBB 1/8
        x_c = self.context_branch_layers[2](self.relu(x_c))
        # x_c = self.layer5(self.relu(x_c))
        x_c = self.spp(x_c)  # DAPPM

        # UNETFormer GLTB
        x_c = self.gltb3(x_c)

        x_c = resize(
            x_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        return (temp_context, x_s + x_c) if self.training else x_s + x_c


def get_pred_model(name, num_classes):
    if 's' in name:
        model = DDRNet1()

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
    # input = torch.randn(1, 3, 1024, 2048).cuda()
    input = torch.randn(1, 3, 512, 1024).cuda()
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




