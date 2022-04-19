from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        bn = nn.BatchNorm2d

        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels,
                          out_channels=self.in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0), bn(self.in_channels))
            # nn.init.constant_(self.W[1].weight, 0)
            # nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels,
                               out_channels=self.in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
            # nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        # 初始化权重
        for m in [self.g, self.theta, self.phi]:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        # print(f.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(self, num_level=4, i_level=2, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.i_level = i_level
        self.num_level = num_level
        self.extra_blocks = extra_blocks

        # ------------------------------- #
        # 定义no-local模块
        # ------------------------------- #
        self.no_local = NonLocalBlockND(in_channels=256, bn_layer=False)

        # ------------------------------- #
        # 定义Ci --> Fi 1x1的卷积层
        # ------------------------------- #
        self.conv_1x1_5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.conv_1x1_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_1x1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)

        # ------------------------------- #
        # 定义Fi --> Pi 3x3的卷积层
        # ------------------------------- #
        self.conv_3x3_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_3x3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_3x3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_3x3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # ------------------------------- #
        # 初始化权重
        # ------------------------------- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.xavier_init(m, distribution='uniform')

    def xavier_init(self, m, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(m.weight, gain=gain)
        else:
            nn.init.xavier_normal_(m.weight, gain=gain)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, bias)

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())

        # ------------------------------- #
        # 先卷积得到Pi
        # ------------------------------- #
        P5 = self.conv_3x3_5(self.conv_1x1_5(x[3]))
        P4 = self.conv_3x3_4(self.conv_1x1_4(x[2]))
        P3 = self.conv_3x3_3(self.conv_1x1_3(x[1]))
        P2 = self.conv_3x3_2(self.conv_1x1_2(x[0]))

        inputs = []
        feats = []
        for i in [P2, P3, P4, P5]:
            inputs.append(i)
        # ------------------------------- #
        # 进行特征融合，得到特征图I
        # ------------------------------- #
        gather_size = inputs[self.i_level].size()[2:]
        for i in range(self.num_level):
            if i < self.i_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        I = sum(feats) / len(feats)

        # ------------------------------- #
        # 进行no-local，得到特征图R
        # ------------------------------- #
        R = self.no_local(I)

        # ------------------------------- #
        # 得到Ri
        # ------------------------------- #

        outs = []
        for i in range(self.num_level):
            out_size = inputs[i].size()[2:]
            if i < self.i_level:
                residual = F.interpolate(R, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(R, output_size=out_size)
            outs.append(residual + inputs[i])

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            outs, names = self.extra_blocks(outs, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, outs)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names
