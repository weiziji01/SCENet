import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer


class LearnableEdgeConv(nn.Module):
    """可学习的边缘提取卷积（替代Sobel、Laplacian固定算子）"""

    def __init__(self, in_channels, kernel_type="x"):
        super().__init__()
        self.depthwise = build_conv_layer(
            cfg=dict(type="Conv2d"),
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self._init_weights(kernel_type)
        self.conv1x1 = build_conv_layer(
            cfg=dict(type="Conv2d"),
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def _init_weights(self, kernel_type):
        if kernel_type == "x":
            kernel = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            )
        elif kernel_type == "y":
            kernel = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            )
        elif kernel_type == "l":
            kernel = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
            )
        kernel = kernel.view(1, 1, 3, 3).repeat(self.depthwise.in_channels, 1, 1, 1)
        self.depthwise.weight.data = kernel

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.depthwise(x)
        return y


class SFM(nn.Module):
    """改进后的上采样模块（替换原始FPN上采样）"""

    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.conv1x1 = build_conv_layer(
            cfg=dict(type="Conv2d"),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv3x3 = build_conv_layer(
            cfg=dict(type="Conv2d"),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )
        self.edge_x = LearnableEdgeConv(self.out_channels, kernel_type="x")
        self.edge_y = LearnableEdgeConv(self.out_channels, kernel_type="y")
        self.edge_conv = build_conv_layer(
            cfg=dict(type="Conv2d"),
            in_channels=self.in_channels * 2,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self.post_aggre_norm = build_norm_layer(dict(type="LN"), self.out_channels)[1]

    def forward(self, x):
        y1 = self.conv3x3(x)
        y2 = self.conv1x1(x)
        y2 = self.conv3x3(y2)
        edge_x = self.edge_x(x)
        edge_y = self.edge_y(x)
        edge = torch.concatenate([edge_x, edge_y], dim=1)
        edge = self.edge_conv(edge)
        aggregated = y1 + y2 + edge + x
        aggregated = aggregated.permute(0, 2, 3, 1)
        aggregated = self.post_aggre_norm(aggregated)
        aggregated = aggregated.permute(0, 3, 1, 2).contiguous()
        return aggregated


class LEM(nn.Module):
    def __init__(self, channels, depth_multiplier=2):
        super(LEM, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.in_channels = channels
        self.out_channels = channels
        self.conv3x3 = build_conv_layer(
            cfg=dict(
                type="Conv2d",
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.conv1x1 = build_conv_layer(
            cfg=dict(type="Conv2d"),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.edge_l = LearnableEdgeConv(self.out_channels, kernel_type="l")
        self.post_enhance_norm = build_norm_layer(dict(type="LN"), self.out_channels)[1]

    def forward(self, x):
        y1 = self.conv3x3(x)
        y2 = self.conv1x1(x)
        y2 = self.conv3x3(y2)
        y3 = self.edge_l(x)
        y = y1 + y2 + y3 + x
        y = y.permute(0, 2, 3, 1)
        y = self.post_enhance_norm(y)
        y = y.permute(0, 3, 1, 2).contiguous()

        return y
