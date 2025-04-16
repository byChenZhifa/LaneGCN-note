# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail


import numpy as np
# from fractions import gcd
from math import gcd

from numbers import Number

import torch
from torch import nn
from torch.nn import functional as F


# Conv layer with norm (gn or bn) and relu.
class Conv(nn.Module):
    """Conv = Conv2D
    2D卷积+归一化+ReLU三件套
    原理：通过组归一化(GN)或批归一化(BN)稳定训练过程
    应用：用于处理二维特征图（如BEV特征）
    """

    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Conv, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv2d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            # self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm = nn.GroupNorm(16, n_out)  # 确保128%16==0
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Conv1d(nn.Module):
    """1D卷积变体，结构与Conv类似: 1D卷积+归一化+ReLU三件套
    应用：处理序列数据（如车辆轨迹）;

    paper:虽然 CNN 和 RNN 都可用于处理时态数据，
    但在这里，我们使用一维 CNN 来处理轨迹输入，因为**它能有效提取多尺度特征并提高并行计算效率。**
    """

    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Conv1d, self).__init__()
        # 卷积层定义
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            # self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm = nn.GroupNorm(16, n_out)  # 确保128%16==0
            
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm="GN", ng=32, act=True):
        super(Linear, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == "GN":
            # self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm = nn.GroupNorm(16, n_out)  # 确保128%16==0
            
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


# Post residual layer
class PostRes(nn.Module):
    """标准残差块（ResNet风格）
    原理：通过跳跃连接解决梯度消失问题，结构：
        x → conv1 → norm1 → relu → conv2 → norm2 → + → relu
        |__________________________________________|
    """

    def __init__(self, n_in, n_out, stride=1, norm="GN", ng=32, act=True):
        super(PostRes, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv1 = nn.Conv2d(
            n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm2d(n_out)
            self.bn2 = nn.BatchNorm2d(n_out)
        else:
            exit("SyncBN has not been added!")

        if stride != 1 or n_out != n_in:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    """1D残差块
    应用：在ActorNet中处理车辆轨迹序列

    Res1d 残差连接示意图：
    Input
    │
    ├─→ Conv1d → BN → ReLU → Conv1d → BN →
    │                                    ⊕
    └────────────────────────────────────┘
                                        │
                                        Output
                                        
    输入形状: [batch, channels, seq_len]
    输出形状: 保持输入形状
    结构：
        Conv1d → GN → ReLU → Conv1d → GN → Add → ReLU
    功能：提取时序特征，残差连接避免梯度消失
    """

    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Res1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        if stride != 1 or n_out != n_in:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    """线性残差块
    原理：将残差思想应用于全连接层
        x → linear1 → norm1 → relu → linear2 → norm2 → + → relu
        |_____________________________________________|
        
    输入形状: [..., features]
    输出形状: 保持特征维度
    结构：
        Linear → GN → ReLU → Linear → GN → Add → ReLU
    功能：全连接层的残差变体，增强特征表达能力
    """

    def __init__(self, n_in, n_out, norm="GN", ng=32):
        super(LinearRes, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == "GN":
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        if n_in != n_out:
            if norm == "GN":
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False), nn.BatchNorm1d(n_out)
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class Null(nn.Module):
    """空操作层（占位符）"""

    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


def linear_interp(x, n_max):
    """双线性插值核心函数
    输入：归一化坐标x ∈ [0,1]，映射到n_max维度的绝对坐标
    输出：左右相邻点的权重和索引
    算法步骤：
        1. 将x从[0,1]映射到[-0.5, n_max-0.5]
        2. 计算左右相邻整数坐标
        3. 根据距离计算权重

    Given a Tensor of normed positions, returns linear interplotion weights and indices.
    Example: For position 1.2, its neighboring pixels have indices 0 and 1, corresponding
    to coordinates 0.5 and 1.5 (center of the pixel), and linear weights are 0.3 and 0.7.

    Args:
        x: Normalizzed positions, ranges from 0 to 1, float Tensor.
        n_max: Size of the dimension (pixels), multiply x to get absolution positions.
    Returns: Weights and indices of left side and right side.
    """
    x = x * n_max - 0.5

    mask = x < 0
    x[mask] = 0
    mask = x > n_max - 1
    x[mask] = n_max - 1
    n = torch.floor(x)

    rw = x - n
    lw = 1.0 - rw
    li = n.long()
    ri = li + 1
    mask = ri > n_max - 1
    ri[mask] = n_max - 1

    return lw, li, rw, ri


def get_pixel_feat(fm, bboxes, pts_range):
    x, y = bboxes[:, 0], bboxes[:, 1]
    x_min, x_max, y_min, y_max = pts_range[:4]
    x = (x - x_min) / (x_max - x_min)
    y = (y_max - y) / (y_max - y_min)

    _, fm_h, fm_w = fm.size()
    xlw, xli, xhw, xhi = linear_interp(x, fm_w)
    ylw, yli, yhw, yhi = linear_interp(y, fm_h)
    feat = (
        (xlw * ylw).unsqueeze(1) * fm[:, yli, xli].transpose(0, 1)
        + (xlw * yhw).unsqueeze(1) * fm[:, yhi, xli].transpose(0, 1)
        + (xhw * ylw).unsqueeze(1) * fm[:, yli, xhi].transpose(0, 1)
        + (xhw * yhw).unsqueeze(1) * fm[:, yhi, xhi].transpose(0, 1)
    )
    return feat


def get_roi_feat(fm, bboxes, roi_size, pts_range):
    """ROI特征提取（Rotated ROI Align）
    原理：根据旋转框参数生成采样网格，通过双线性插值获取特征
    流程：
        1. 根据bbox参数生成采样点网格（考虑旋转角度）
        2. 将采样点坐标归一化到特征图范围
        3. 使用linear_interp进行双线性插值
        4. 聚合有效区域的特征
    应用：用于从BEV特征图中提取旋转区域特征

    Rotated ROI Align 流程：
        1. 定义旋转框参数 (cx, cy, w, h, θ)
        2. 生成采样网格（考虑旋转角度）
        ┌──────────────┐
        │  bbox中心    │
        │    ┌───────┐ │
        │    │  ROI  │ │ → 通过旋转矩阵计算采样点
        │    └───────┘ │
        └──────────────┘
        3. 双线性插值获取特征值


    Given a set of BEV bboxes get their BEV ROI features.

    Args:
        fm: Feature map, float tensor, chw
        bboxes: BEV bboxes, n x 5 float tensor (cx, cy, wid, hgt, theta)
        roi_size: ROI size (number of bins), [int] or int
        pts_range: Range of points, tuple of ints, (x_min, x_max, y_min, y_max, z_min, z_max)
    Returns: Extracted features of size (num_roi, c, roi_size, roi_size).
    """
    if isinstance(roi_size, Number):
        roi_size = [roi_size, roi_size]

    cx, cy, wid, hgt, theta = (
        bboxes[:, 0],
        bboxes[:, 1],
        bboxes[:, 2],
        bboxes[:, 3],
        bboxes[:, 4],
    )
    st = torch.sin(theta)
    ct = torch.cos(theta)
    num_bboxes = len(bboxes)

    # 生成旋转采样网格
    rot_mat = bboxes.new().resize_(num_bboxes, 2, 2)
    rot_mat[:, 0, 0] = ct  # 根据角度计算旋转矩阵
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct

    # 生成相对采样点
    offset = bboxes.new().resize_(len(bboxes), roi_size[0], roi_size[1], 2)
    x_bin = (torch.arange(roi_size[1]).float().to(bboxes.device) + 0.5) / roi_size[
        1
    ] - 0.5
    offset[:, :, :, 0] = x_bin.view(1, 1, -1) * wid.view(-1, 1, 1)
    y_bin = (
        torch.arange(roi_size[0] - 1, -1, -1).float().to(bboxes.device) + 0.5
    ) / roi_size[0] - 0.5
    offset[:, :, :, 1] = y_bin.view(1, -1, 1) * hgt.view(-1, 1, 1)

    rot_mat = rot_mat.view(num_bboxes, 1, 1, 2, 2)
    offset = offset.view(num_bboxes, roi_size[0], roi_size[1], 2, 1)
    offset = torch.matmul(rot_mat, offset).view(num_bboxes, roi_size[0], roi_size[1], 2)

    # 应用旋转矩阵后的绝对坐标
    x = cx.view(-1, 1, 1) + offset[:, :, :, 0]
    y = cy.view(-1, 1, 1) + offset[:, :, :, 1]
    x = x.view(-1)
    y = y.view(-1)

    x_min, x_max, y_min, y_max = pts_range[:4]
    x = (x - x_min) / (x_max - x_min)
    y = (y_max - y) / (y_max - y_min)

    fm_c, fm_h, fm_w = fm.size()
    feat = fm.new().float().resize_(num_bboxes * roi_size[0] * roi_size[1], fm_c)
    mask = (x > 0) * (x < 1) * (y > 0) * (y < 1)
    x = x[mask]
    y = y[mask]

    xlw, xli, xhw, xhi = linear_interp(x, fm_w)
    ylw, yli, yhw, yhi = linear_interp(y, fm_h)
    feat[mask] = (
        (xlw * ylw).unsqueeze(1) * fm[:, yli, xli].transpose(0, 1)
        + (xlw * yhw).unsqueeze(1) * fm[:, yhi, xli].transpose(0, 1)
        + (xhw * ylw).unsqueeze(1) * fm[:, yli, xhi].transpose(0, 1)
        + (xhw * yhw).unsqueeze(1) * fm[:, yhi, xhi].transpose(0, 1)
    )
    feat[torch.logical_not(mask)] = 0
    feat = feat.view(num_bboxes, roi_size[0] * roi_size[1], fm_c)
    feat = (
        feat.transpose(1, 2).contiguous().view(num_bboxes, -1, roi_size[0], roi_size[1])
    )
    return feat
