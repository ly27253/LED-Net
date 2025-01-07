# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


@MODELS.register_module()
class LEDHead(BaseDecodeHead):
    """Decode head for LEDNet.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        num_classes (int): Number of classes.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)
        self.aux_head = self._make_base_head(self.in_channels // 2,
                                             self.channels)
        self.head_x1 = self._make_base_head(32, 2)
        self.head_x2 = self._make_base_head(32, 2)

        self.aux_cls_seg = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            inputs: Union[Tensor,
                          Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            c3_feat, c5_feat, x1, x2 = inputs
            x_c = self.head(c5_feat)   # ly修改
            x_c = self.cls_seg(x_c)   # ly修改
            x_s = self.aux_head(c3_feat)
            x_s = self.aux_cls_seg(x_s)
            head_x1 = self.head_x1(x1)   # ly修改
            head_x2 = self.head_x2(x2)   # ly修改

            return x_c, x_s, head_x1, head_x2   # ly修改
        else:
            head_x1 = self.head_x1(inputs[1])   # ly修改
            head_x2 = self.head_x2(inputs[2])   # ly修改
            x_c = self.head(inputs[0])
            x_c = self.cls_seg(x_c)
            return (x_c, head_x1, head_x2)   # ly修改
            # return x_c

    def _make_base_head(self, in_channels: int,
                        channels: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('norm', 'act', 'conv')),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]

        return nn.Sequential(*layers)

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        context_logit, spatial_logit, head_x1, head_x2 = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)
        context_logit = head_x2 + resize(
        # context_logit = resize(
            context_logit,
            size=tuple(s // 4 for s in seg_label.shape[2:]),
            mode='bilinear',
            align_corners=self.align_corners)
        context_logit = head_x1 + resize(
            context_logit,
            size=tuple(s // 2 for s in seg_label.shape[2:]),
            mode='bilinear',
            align_corners=self.align_corners)
        context_logit = resize(
            context_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        spatial_logit = head_x2 + resize(
        # spatial_logit = resize(
            spatial_logit,
            size=tuple(s // 4 for s in seg_label.shape[2:]),
            mode='bilinear',
            align_corners=self.align_corners)
        spatial_logit = head_x1 + resize(
            spatial_logit,
            size=tuple(s // 2 for s in seg_label.shape[2:]),
            mode='bilinear',
            align_corners=self.align_corners)
        spatial_logit = resize(
            spatial_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)

        loss['loss_context'] = self.loss_decode[0](context_logit, seg_label)
        loss['loss_spatial'] = self.loss_decode[1](spatial_logit, seg_label)
        loss['acc_seg'] = accuracy(
            context_logit, seg_label, ignore_index=self.ignore_index)

        return loss
