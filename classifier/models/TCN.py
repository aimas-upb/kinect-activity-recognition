import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))


class Unit_brdc(nn.Module):
    def __init__(self, D_in, D_out, kernel_size, stride=1, dropout=0):

        super(Unit_brdc, self).__init__()
        self.bn = nn.BatchNorm1d(D_in)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            D_in,
            D_out,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=stride)

        # weight initialization
        conv_init(self.conv)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv(x)
        return x


class TCN_unit(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=9, stride=1):
        super(TCN_unit, self).__init__()
        self.unit1_1 = Unit_brdc(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            dropout=0.5,
            stride=stride)

        if in_channel != out_channel:
            self.down1 = Unit_brdc(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        x =self.unit1_1(x)\
                + (x if self.down1 is None else self.down1(x))
        return x


class TCN1DModel(nn.Module):
    def __init__(self, channel, num_class, joint_size, no_of_joints, use_data_bn=False):
        super(TCN1DModel, self).__init__()
        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.joint_size = joint_size
        self.no_of_joints = no_of_joints
        self.data_bn = nn.BatchNorm1d(channel)
        self.conv0 = nn.Conv1d(channel, 128, kernel_size=13, padding=6)
        self.conv1 = nn.Conv1d(128, 64, kernel_size=9, padding=4)
        conv_init(self.conv0)
        conv_init(self.conv1)

        self.unit1 = TCN_unit(64, 64)
        self.unit2 = TCN_unit(64, 64)
        self.unit3 = TCN_unit(64, 64)
        self.unit4 = TCN_unit(64, 128, stride=2)
        self.unit5 = TCN_unit(128, 128)
        self.unit6 = TCN_unit(128, 128)
        self.unit7 = TCN_unit(128, 256, stride=2)
        self.unit8 = TCN_unit(256, 256)
        self.unit9 = TCN_unit(256, 256)
        self.unit10 = TCN_unit(256, 512, stride=2)
        self.unit11 = TCN_unit(512, 512)
        self.unit12 = TCN_unit(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        self.fcn0 = nn.Conv1d(512, 256, kernel_size=1)
        self.fcn1 = nn.Conv1d(256, 128, kernel_size=1)
        self.fcn2 = nn.Conv1d(128, num_class, kernel_size=1)
        conv_init(self.fcn0)
        conv_init(self.fcn1)
        conv_init(self.fcn2)

    def forward(self, x, samples):
        t = x.size()
        x = x.view(t[0], t[1], -1, self.joint_size)
        x = x[:, :, :, 0:3]
        x = x.transpose(1, 2).contiguous()

        x = torch.unsqueeze(x, 4)
        x = x.transpose(1, 3).contiguous()
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        if self.use_data_bn:
            x = self.data_bn(x)
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        x = self.unit4(x)
        x = self.unit5(x)
        x = self.unit6(x)
        x = self.unit7(x)
        x = self.unit8(x)
        x = self.unit9(x)
        x = self.unit10(x)
        x = self.unit11(x)
        x = self.unit12(x)
        x = self.bn(x)
        x = self.relu(x)

        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        x = self.fcn0(x)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = x.view(-1, self.num_class)

        return x