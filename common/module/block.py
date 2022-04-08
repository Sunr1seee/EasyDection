import math
import torch
import torch.nn as nn

from common.register import *

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import warnings

import torch
import torch.nn as nn

from common.register import *
from layer import *


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


@BLOCK.register()
class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, ci, co, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if ci != co:
            self.conv = Conv(ci, co)
        self.linear = nn.Linear(co, co)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(co, num_heads) for _ in range(num_layers)))
        self.co = co

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.co, w, h)


@BLOCK.register()
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, ci, co, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(co * e)  # hidden channels
        self.cv1 = Conv(ci, c_, 1, 1)
        self.cv2 = Conv(c_, co, 3, 1, g=g)
        self.add = shortcut and ci == co

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@BLOCK.register()
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, ci, co, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(co * e)  # hidden channels
        self.cv1 = Conv(ci, c_, 1, 1)
        self.cv2 = nn.Conv2d(ci, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, co, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@BLOCK.register()
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, ci, co, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(co * e)  # hidden channels
        self.cv1 = Conv(ci, c_, 1, 1)
        self.cv2 = Conv(ci, c_, 1, 1)
        self.cv3 = Conv(2 * c_, co, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


@BLOCK.register()
class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, ci, co, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(ci, co, n, shortcut, g, e)
        c_ = int(co * e)
        self.m = TransformerBlock(c_, c_, 4, n)


@BLOCK.register()
class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, ci, co, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(ci, co, n, shortcut, g, e)
        c_ = int(co * e)
        self.m = SPP(c_, c_, k)


@BLOCK.register()
class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, ci, co, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(ci, co, n, shortcut, g, e)
        c_ = int(co * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


@BLOCK.register()
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, ci, co, k=(5, 9, 13)):
        super().__init__()
        c_ = ci // 2  # hidden channels
        self.cv1 = Conv(ci, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), co, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


@BLOCK.register()
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, ci, co, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = ci // 2  # hidden channels
        self.cv1 = Conv(ci, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, co, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


@BLOCK.register()
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, ci, co, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = co // 2  # hidden channels
        self.cv1 = Conv(ci, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


@BLOCK.register()
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, ci, co, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = co // 2
        self.conv = nn.Sequential(GhostConv(ci, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, co, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(ci, ci, k, s, act=False),
                                      Conv(ci, co, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
