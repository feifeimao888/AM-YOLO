import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




class E_SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.cv2 = Conv(c_ * 3, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.a = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y0 = self.cv1(x)
        y1 = self.a(y0)
        y2 = self.m(y1)
        # y3 = self.m(y2)

        # all = torch.cat(y0,y1,y2,y3, 1)
        all = torch.cat((y0, y1, y2), dim=1)
        end = self.cv2(all)

        return end