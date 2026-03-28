import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Stem(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, relu=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicStage(nn.Module):
    def __init__(self, dim_list, max_pool=False):
        super(BasicStage, self).__init__()
        self.blocks = []
        if max_pool:
            self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        for i in range(len(dim_list) - 1):
            block = BasicBlock(dim_list[i], dim_list[i + 1], kernel_size=3 if i % 2 == 0 else 1, stride=1,
                               padding=1 if i % 2 == 0 else 0)
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.stem = Stem(3, 32, kernel_size=3, stride=2, padding=1)
        self.s1 = BasicStage([32, 64, 32, 64])
        self.s2 = BasicStage([64, 128, 64, 128], max_pool=True)
        self.s3 = BasicStage([128, 208, 128, 208, 128, 208], max_pool=True)
        self.s4 = nn.Sequential(
            BasicStage([208, 256, 208, 256, 208, 256], max_pool=True),
            BasicBlock(256, 256, 3, 1, 1)
        )
        self.s4_2 = nn.Sequential(
            BasicBlock(208, 128, 3, 1, 1),
            BasicBlock(128, 512, 3, 2, 1)
        )
        self.s5 = nn.Sequential(
            BasicBlock(768, 256, 3, 1, 1),
            BasicBlock(256, 256, 3, 1, 1)
        )
        self.fusion_s1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.fusion_s2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1)
        )
        self.fusion_s3 = nn.Sequential(
            nn.Conv2d(208, 64, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )
        self.fusion_s5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        x4_2 = self.s4_2(x3)
        x5 = self.s5(torch.cat([x4_2, x4], 1))

        # 8x混合
        x = self.fusion_s1(x1) + self.fusion_s2(x2) + self.fusion_s3(x3) + self.fusion_s5(x5)
        x = self.head(x)
        return x


class DCEPredictor():
    """
    predictor of DCE representation and DensePose 15 coarse segmentation (14 parts / background)
    """

    def __init__(self, dim_in, s_chan, p_chan, dce_chan):
        super(DCEPredictor, self).__init__()
        self.s_chan = s_chan
        self.p_chan = p_chan
        self.dce_chan = dce_chan
        self.decode = BasicBlock(
            dim_in,
            s_chan + p_chan + dce_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            relu=False
        )

    def interp2d(self, size):
        """
        Args:
            tensor_nchw: shape (N, C, H, W)
        Return:
            tensor of shape (N, C, Hout, Wout) by applying the scale factor to H and W
        """
        return nn.functional.interpolate(
            size, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )

    def forward(self, head_outputs):
        x = F.interpolate(head_outputs, scale_factor=2, mode='nearest')
        x = self.decode(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        s_out, p_out, dce_out = torch.split(x, [self.s_chan, self.p_chan, self.dce_chan], dim=1)
        dce_out = F.normalize(dce_out, 1)
        return s_out, p_out, dce_out


class DCEModel(nn.Module):
    def __init__(self, s_chan, p_chan, dce_chan):
        super(DCEModel, self).__init__()
        self.backbone = DarkNet()
        self.DCEPredictor = DCEPredictor(dim_in=256, s_chan=s_chan, p_chan=p_chan, dce_chan=dce_chan)
        self.loss = ContinuousSurfaceEmbeddingLoss()
