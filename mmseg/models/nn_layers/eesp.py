from torch.nn import init
import torch.nn.functional as F
from ..nn_layers.espnet_utils import *
import math
import torch
from ..classification import espnetv2_config as config

#============================================
__author__ = " "
__maintainer__ = " "
#============================================

config_inp_reinf = config.config_inp_reinf

class SESP(nn.Module):
    '''
    最新更新：2024年6月27日，增加了d_rate的判断
    This class defines the SESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp', Spatial=True, SPASPP_Flag=False, SESPV2=True):  #down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for SESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        self.Spatial = Spatial
        self.SESPV2 = SESPV2
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)
        self.k_sizes = list()
        if self.Spatial:
            map_receptive_ksize = {3: 1, 5: 1, 7: 1, 9: 1, 11: 1, 13: 1, 15: 1, 17: 1}
            for i in range(k):
                ksize = int(3)
                self.k_sizes.append(ksize)
        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        else:
            map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 6, 19: 12, 21: 18, 23: 24}
            for i in range(k):
                ksize = int(3 + 2 * i)
                # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
                ksize = ksize if ksize <= r_lim else 3
                self.k_sizes.append(ksize)
            # sort (in ascending order) these kernel sizes based on their receptive field
            # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
            # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
            self.k_sizes.sort()
        if SPASPP_Flag:
            self.k_sizes = [17, 19, 21, 23]  # 如果使能，则使用SPASPP的大空洞率 0709
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
        if self.SESPV2:
            self.spp_dw_v2 = nn.ModuleList()
            for i in range(k):
                d_rate = map_receptive_ksize[self.k_sizes[i]]
                self.spp_dw_v2.append(CDilated(n, n, kSize=3, stride=1, groups=n, d=d_rate+1))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        if self.SESPV2:   # 在结构之后进行
            output_v2 = [self.spp_dw_v2[0](output[0])]
            for k in range(1, len(self.spp_dw)):
                out_k = self.spp_dw_v2[k](output[k])
                output_v2.append(out_k)
            output = output_v2
            del output_v2
        # Merge
        expanded = self.conv_1x1_exp( # learn linear combinations using group point-wise convolutions
            self.br_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1) # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded
        elif self.stride == 2 and self.Spatial==False:
            return expanded + self.avg(input)

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)

class EEESP(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'): #down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp( # learn linear combinations using group point-wise convolutions
            self.br_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1) # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:   # 增加跳连如何
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)

class DownSampler(nn.Module):
    '''
    Down-sampling fucntion that has three parallel branches: (1) avg pooling,
    (2) EESP block with stride of 2 and (3) efficient long-range connection with the input.
    The output feature maps of branches from (1) and (2) are concatenated and then additively fused with (3) to produce
    the final output.
    '''

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True, Keep_channels_up=True, Keep_scale=False, channels_down=False):
        '''
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        '''
        super().__init__()
        self.Keep_channels_up = Keep_channels_up
        self.Keep_scals = Keep_scale
        self.channels_down = channels_down
        if self.Keep_channels_up and nout >= nin:
            nout_new = nout - nin
        elif self.Keep_channels_up and nout < nin:
            nout_new = nin - nout
        else:
            nout_new = nout
        if self.Keep_scals:
            self.stride = 1
        else:
            self.stride = 2
        self.eesp = EESP(nin, nout_new, stride=self.stride, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=self.stride)
        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(config_inp_reinf, config_inp_reinf, 3, 1),
                CB(config_inp_reinf, nout, 1, 1)
            )
        if self.channels_down:
            self.CBR_down = CBR(nin, nout, 3, 1)
        self.act =  nn.PReLU(nout)

    def forward(self, input, input2=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        avg_out = self.avg(input)
        if self.channels_down:
            avg_out = self.CBR_down(avg_out)
        eesp_out = self.eesp(input)
        if self.Keep_channels_up:
            output = torch.cat([avg_out, eesp_out], 1)
        else:
            output = avg_out + eesp_out
        if input2 is not None:
            #assuming the input is a square image
            # Shortcut connection with the input image
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.act(output)