import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity
from thop import profile

##########################################################################################################################
"""
【my】
【高阶 低阶 都上升】
加Bn

最后用加法
"""

class CSFblock(nn.Module):
    def __init__(self, high_channels, low_channels,outChannels):
        super().__init__()

        self.UpChannels = nn.Sequential(
            nn.Conv2d(low_channels, outChannels, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(outChannels),)
        self.DownChannels = nn.Sequential(
            nn.Conv2d(high_channels, outChannels, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(outChannels),)

        self.AdaptiveAvgPool2d_1 = nn.AdaptiveAvgPool2d(1)
        self.AdaptiveAvgPool2d_2 = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, inputs):  # 接收包含多个输入的列表
        # 解包输入列表（假设输入是 [x_h, x_l]）
        if len(inputs) != 2:
            raise ValueError("CSFblock requires exactly 2 input tensors.")
        High, Low = inputs[0], inputs[1]  # 提取两个输入张量
        #x1 = torch.Size([1, 256, 40, 40])
        #x2 = torch.Size([1, 128, 40, 40])

        N_High = self.DownChannels(High)
        N_Low = self.UpChannels(Low)

        key_H = self.AdaptiveAvgPool2d_1(N_High)
        key_L = self.AdaptiveAvgPool2d_2(N_Low)
        """x_se1和x_se2的值是不同的，它们拥有独立的权重参数"""

        key_twice = torch.cat([key_H, key_L], 2)

        softKey = self.softmax(key_twice)

        doorH = torch.unsqueeze(softKey[:, :, 0], 2)
        doorL = torch.unsqueeze(softKey[:, :, 1], 2)

        good_high = doorH * N_High
        good_low = doorL * N_Low

        well_dool = good_high+ good_low

        return well_dool

if __name__ == "__main__":

    xin1 = torch.randn(1, 256, 40, 40)
    xin2 = torch.randn(1, 128, 40, 40)
    input_ = [xin1, xin2]

    model = CSFblock(256,128,384)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #flops, _ = profile(model, inputs=(input_))

    with torch.no_grad():
        output = model(input_)
        output_shape = output.shape

    print(f" {params:,}")
    #print(f"计算量(FLOPs): {flops:,}")
    print(f" {output_shape}")


