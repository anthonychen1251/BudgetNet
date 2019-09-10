# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor, block_id):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 16.)
        self.path = []
        self.block_id = block_id
        self.block_policy_idx = [block_id*cardinality, (block_id+1)*cardinality]
        path_channel = int(base_width * width_ratio)
        for i in range(cardinality):
            path_module = nn.Sequential(OrderedDict([
                ('conv_reduce', nn.Conv2d(in_channels, path_channel, kernel_size=1, stride=1, padding=0, bias=False)),
                ('bn_reduce', nn.BatchNorm2d(path_channel)),
                ('relu_reduce', nn.ReLU(inplace=True)),
                ('conv_conv', nn.Conv2d(path_channel, path_channel, kernel_size=3, stride=stride, padding=1, bias=False)),
                ('bn', nn.BatchNorm2d(path_channel)),
                ('relu_conv', nn.ReLU(inplace=True)),
                ('conv_expand', nn.Conv2d(path_channel, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            ]))
            self.path.append(path_module)
        self.path = nn.ModuleList(self.path)
        self.bn = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x, policy):
        available_policy = policy[:, self.block_policy_idx[0]:self.block_policy_idx[1]]
        for index, layer in enumerate(self.path):
            mask = available_policy[:, index].float().view(-1,1,1,1)
            if index==0:
                bottleneck = layer.forward(x) * mask
            else:
                bottleneck += layer.forward(x) * mask
        bottleneck = self.bn(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)

    def forward_single(self, x, policy):
        available_policy = policy[self.block_policy_idx[0]:self.block_policy_idx[1]]
        residual = self.shortcut.forward(x)
        bottleneck = torch.zeros(residual.shape).cuda()
        for index, layer in enumerate(self.path):
            mask = available_policy[index].float()
            if mask.item()==1:                
                bottleneck += layer.forward(x)
            else:
                continue

        
        bottleneck = self.bn(bottleneck)
        return F.relu(residual + bottleneck, inplace=True)

    def forward_all(self, x):
        residual = self.shortcut.forward(x)
        bottleneck = torch.zeros(residual.shape).cuda()
        for index, layer in enumerate(self.path):
            bottleneck += layer.forward(x)
        bottleneck = self.bn(bottleneck)
        return F.relu(residual + bottleneck, inplace=True)

class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stage_policy_num = [self.block_depth * self.cardinality  * (i+1) for i in range(2)]
        edited_base = 16
        self.stages = [edited_base, edited_base * self.widen_factor, edited_base* 2 * self.widen_factor, edited_base * 4 * self.widen_factor]
        #default 64
        self.conv_1_3x3 = nn.Conv2d(3, edited_base, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(edited_base)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal_(self.classifier.weight)
        for key in self.state_dict():
#            print(key)
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = []
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.append(ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor, bottleneck))
            else:
                block.append(ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor, bottleneck))
        block = nn.ModuleList(block)
        return block


    def forward(self, x, policy):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        for i, l in enumerate(self.stage_1):
            x = self.stage_1[i](x, policy[:, 0:self.stage_policy_num[0]])
        for i, l in enumerate(self.stage_2):
            x = self.stage_2[i](x, policy[:, self.stage_policy_num[0]:self.stage_policy_num[1]])
        for i, l in enumerate(self.stage_3):
            x = self.stage_3[i](x, policy[:, self.stage_policy_num[1]:])
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)

    def forward_single(self, x, policy):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        for i, l in enumerate(self.stage_1):
            x = self.stage_1[i].forward_single(x, policy[0:self.stage_policy_num[0]])
        for i, l in enumerate(self.stage_2):
            x = self.stage_2[i].forward_single(x, policy[self.stage_policy_num[0]:self.stage_policy_num[1]])
        for i, l in enumerate(self.stage_3):
            x = self.stage_3[i].forward_single(x, policy[self.stage_policy_num[1]:])
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)

    def forward_all(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        for i, l in enumerate(self.stage_1):
            x = self.stage_1[i].forward_all(x)
        for i, l in enumerate(self.stage_2):
            x = self.stage_2[i].forward_all(x)
        for i, l in enumerate(self.stage_3):
            x = self.stage_3[i].forward_all(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)

