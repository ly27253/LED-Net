#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

from torch import nn
import math
from ..nn_layers.cnn_utils import CBR

import torch

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]  ##rows
    out_cols = (cols + strides[1] - 1) // strides[1]  ##cols
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1  ##3
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1  ##3
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)  ##2
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)  ##2
    # Pad the input
    padding_top = int(padding_rows / 2.)  ##1
    padding_left = int(padding_cols / 2.)  ##1
    padding_bottom = padding_rows - padding_top  ##1
    padding_right = padding_cols - padding_left  ##1
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = nn.ZeroPad2d(paddings)(images)
    return images

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# MLP in the paper
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim // 2,
                                bias=qkv_bias)  # nn.Linear方法接受输入参数作为权重，需要将4维张量转换为权重后才能进行输入”
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_all = torch.split(q, math.ceil(N // 4), dim=-2)
        k_all = torch.split(k, math.ceil(N // 4), dim=-2)
        v_all = torch.split(v, math.ceil(N // 4), dim=-2)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale  # 16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)  # .reshape(B, N, C)
            output.append(trans_x)
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x


'''
# Efficient Multi-Head Attention in the paper
class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or 1//4##head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim//9, bias=qkv_bias) 
        self.qkv = nn.Linear(dim//9, dim * 3//9, bias=qkv_bias)
        self.proj = nn.Linear(dim//9, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
'''


## Key Module: Efficient Transformer (ET) in the paper
class TransBlock(nn.Module):
    def __init__(
            self, n_feat=32, dim=288, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(TransBlock, self).__init__()
        self.dim = dim
        self.atten = EffAttention(self.dim, num_heads=8, qkv_bias=False, qk_scale=None,
                                  attn_drop=0., proj_drop=0.)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim // 4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x):
        B = x.shape[0]
        x = extract_image_patches(x, ksizes=[3, 3],
                                  strides=[1, 1],
                                  rates=[1, 1],
                                  padding='same')  # 16*2304*576
        x = x.permute(0, 2, 1)

        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientPWConv(nn.Module):
    def __init__(self, nin, nout):
        super(EfficientPWConv, self).__init__()
        self.wt_layer = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                        nn.Sigmoid()
                    )

        self.groups = math.gcd(nin, nout)
        self.expansion_layer = CBR(nin, nout, kSize=3, stride=1, groups=self.groups)

        self.out_size = nout
        self.in_size = nin

    def forward(self, x):
        wts = self.wt_layer(x)
        x = self.expansion_layer(x)
        x = x * wts
        return x

    def __repr__(self):
        s = '{name}(in_channels={in_size}, out_channels={out_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

