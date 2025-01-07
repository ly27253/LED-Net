# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule, Sequential

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from mmseg.models.backbones.stdc import STDCModule
from mmseg.models.backbones.UNetFormer_GETB import GETBBlock
from .model_utils import MSFA, CSESP, SPASPP
from ..nn_layers.eesp import SESP

def visualize_features(features, layer_name):
    """
    可视化第三个通道的特征图并进行2倍上采样
    :param features: 要可视化的特征图
    :param layer_name: 图层的名称，用于标题
    """
    # 假设特征图的形状为 [batch_size, channels, height, width]
    feature_map = features[0]  # 选择第一个样本
    third_channel = feature_map[0]  # 选择第三个通道

    # 对特征图进行2倍上采样
    resized_feature_map = F.interpolate(third_channel.unsqueeze(0).unsqueeze(0), scale_factor=2, mode='bilinear',
                                        align_corners=False).squeeze(0).squeeze(0)

    # 创建一个窗口来显示特征图
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_feature_map.cpu().detach().numpy(), cmap='viridis')  # 使用合适的颜色映射
    plt.axis('off')  # 去除坐标轴
    plt.suptitle(f'Feature Map from {layer_name} - Channel 3', fontsize=16)
    plt.show()

# 可视化 x_c 和 x_s
def visualize_xc_xs(x_c, x_s):
    """
    可视化 x_c 和 x_s
    :param x_c: context 特征图
    :param x_s: spatial 特征图
    """
    # 可视化 x_c
    visualize_features(x_c, 'x_c')

    # 可视化 x_s
    visualize_features(x_s, 'x_s')

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
class LEDNet(BaseModule):
    """LEDNet backbone.

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
        self.K = [4, 4, 4, 4, 4]
        self.recept_limit = [13, 11, 9, 7, 5]
        self.eesp_channels = [16, 32, 64, 128, 256, 512, 1024]
        self.num_convs = num_convs
        self.channels = (64, 128, 256, 512, 1024)
        self.channels_spatial = (64, 64, 64, 128, 256)
        self.getb1 = GETBBlock(dim=128, num_heads=8, window_size=8)
        self.getb3 = GETBBlock(dim=128, num_heads=8, window_size=8)

        self.aff1 = MSFA(channels=channels * 2)
        self.aff2 = MSFA(channels=channels * 2)

        self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32, requires_grad=False).reshape((1, 1, 3, 3)).cuda()
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
            dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)
        self.boundary_threshold = 0.1

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

        self.conv_11 = ConvModule(
            3,
            1,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=None, padding=1)
        self.conv_21 = ConvModule(
            1,
            3,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=None, padding=1)

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

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

        self.spatial_branch_layers = nn.ModuleList()
        for i in range(2):
            self.spatial_branch_layers.append(self._make_stdc_eesp_stage(self.channels_spatial[i], self.channels_spatial[i + 1]))

        self.layer3 = nn.Sequential(
                CSESP(planes * 1, planes * 1, planes * 2, stride_1=2, Ker=[4, 4, 4], recept_limit=[13, 11, 9]),
        )

        self.layer4 = nn.Sequential(
                CSESP(planes * 2, planes * 2, planes * 4, stride_1=2, Ker=[4, 4, 4], recept_limit=[13, 11, 9]),
        )

        self.layer5 = SESP(self.eesp_channels[4], self.eesp_channels[3], stride=2, k=self.K[2], r_lim=self.recept_limit[2])
        self.layer5_ = SESP(self.eesp_channels[2], self.eesp_channels[3], stride=1, k=self.K[2], r_lim=self.recept_limit[2], Spatial=False)

        # self.spp = DAPPM(
        #     channels * 16, ppm_channels, channels * 4, num_scales=5)
        # self.spp = DAPPM(512, 128, 128, num_scales=5)   # DDRNet
        # self.spp_end = SPASPP(512, 128, 128)  # DSNet
        # self.spp_end = EESPSPASPP(512, 512, 128)

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

    def _make_stdc_eesp_stage(self, in_planes, out_planes, stride_1=2):
        layers = []
        layers.append(
            CSESP(in_planes, in_planes, out_planes, stride_1=1, Ker=[4, 4, 4], Spatial=True, recept_limit=[13, 11, 9]))
        return Sequential(*layers)

    def _laplacian(self, x):
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

        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0

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

        return boudary_targets_pyramid.float()

    # Spatial Edge Attention Module
    def _laplacian1(self, x):
        seg_label = self.conv_11(x)
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

        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0


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

        # visualize_xc_xs(boundary_targets_x2_up, boundary_targets_x4_up)
        # visualize_xc_xs(boundary_targets, boudary_targets_pyramid.float())

        return boudary_targets_pyramid.float()

    def forward(self, x):
        """Forward function."""
        # out_size = (x.shape[-2] // 8, x.shape[-1] // 8)
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))  # 0527xiugai 向上取zheng
        seg_labels1 = self._laplacian1(x)

        # stage 0-2
        # x = self.stem(x)   # 1/8
        x1 = self.stem[0](x)   # 1/2
        x2 = self.stem[3](self.stem[2](self.stem[1](x1)))   # 1/4
        x = self.stem[5](self.stem[4](x2))   # 1/8

        # visualize_xc_xs(x1, x)

        seg_labels = self._laplacian(x)  # 计算laplas算子

        # stage3
        # x_c = self.context_branch_layers[0](x)
        x_c = self.layer3[0](x)
        # UNETFormer GLTB   128*32*64
        x_c = self.getb1(x_c)

        x_s = self.spatial_branch_layers[0](x)
        comp_c = self.compression_aff(self.relu(x_c))
        x_c = x_c + self.down_1(self.relu(x_s))

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
        x_c = self.layer4(self.relu(x_c))
        x_s = self.spatial_branch_layers[1](self.relu(x_s))
        comp_c = self.compression_2(self.relu(x_c))
        x_c = x_c + self.down_2(self.relu(x_s))

        # visualize_xc_xs(x_c, x_s)

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
        x_s = self.layer5_(self.relu(x_s))   # RBB 1/8
        x_c = self.layer5(self.relu(x_c))
        # x_c = self.spp(x_c)  # DAPPM
        # x_c = self.spp_end(x_c)  # SPASPP

        # UNETFormer GLTB
        x_c = self.getb3(x_c)

        x_c = resize(
            x_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        # return (temp_context, x_s + x_c, x1, x2) if self.training else (x_s + x_c, x1, x2)
        return (temp_context, x_s + x_c, x1, x2) if self.training else (x_s + x_c, x1, x2)