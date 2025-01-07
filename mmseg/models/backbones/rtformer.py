import torch
import torch.nn as nn
import torch.nn.functional as F

def bn2d(in_channels, bn_mom=0.1, lr_mult=1.0, **kwargs):
    assert 'bias_attr' not in kwargs, "bias_attr must not in kwargs"
    param_attr = nn.Parameter(learning_rate=lr_mult)
    return nn.BatchNorm2d(
        in_channels,
        momentum=bn_mom,
        weight_attr=param_attr,
        bias_attr=param_attr,
        **kwargs)
