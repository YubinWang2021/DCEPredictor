import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from loss.dce_loss import DenseCorrepondenceEmbeddingLoss
import thop
from thop import profile

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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.inc = Stem(3, 64,stride=2, padding=1)
        self.down1 = (Down(64, 128))
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 256)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feat = self.outc(x)
        return feat




class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_rate=6, se=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_rate = expansion_rate
        self.se = se

        expansion_channels = in_channels * expansion_rate
        se_channels = max(1, int(in_channels * 0.25))

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            padding = 0

        if expansion_rate != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU()
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=expansion_channels, bias=False),
            nn.BatchNorm2d(expansion_channels),
            nn.ReLU()
        )

        if se:
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(expansion_channels, se_channels, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(se_channels, expansion_channels, 1, bias=False),
                nn.Sigmoid()
            )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expansion_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):
        x = inputs
        if self.expansion_rate != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se:
            x = self.se_block(x) * x
        x = self.pointwise_conv(x)
        if self.in_channels == self.out_channels and self.stride == 1:
            x = x + inputs
        return x

class EffUNet(nn.Module):
    def __init__(self, in_channels=3, classes=256):
        super().__init__()

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

        self.down_block_2 = nn.Sequential(
            MBConvBlock(32, 16, 3, 1, expansion_rate=1),
            MBConvBlock(16, 24, 3, 2),
            MBConvBlock(24, 24, 3, 1),
        )

        self.down_block_3 = nn.Sequential(
            MBConvBlock(24, 40, 5, 2),
            MBConvBlock(40, 40, 5, 1),
        )

        self.down_block_4 = nn.Sequential(
            MBConvBlock(40, 80, 3, 2),
            MBConvBlock(80, 80, 3, 1),
            MBConvBlock(80, 80, 3, 1),
            MBConvBlock(80, 112, 5, 1),
        )

        self.down_block_5 = nn.Sequential(
            MBConvBlock(112, 112, 5, 1),
            MBConvBlock(112, 112, 5, 1),
            MBConvBlock(112, 160, 5, 2),
            MBConvBlock(160, 160, 5, 1),
            MBConvBlock(160, 160, 5, 1),
            MBConvBlock(160, 160, 5, 1),
            MBConvBlock(160, 240, 3, 1),
        )

        self.up_block_4 = DecoderBlock(352, 192)
        self.up_block_3 = DecoderBlock(232, 96)
        self.up_block_2 = DecoderBlock(120, 48)
        self.up_block_1a = DecoderBlock(80, 32)

        self.outc = nn.Conv2d(32, classes, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down_block_2(x1)
        x3 = self.down_block_3(x2)
        x4 = self.down_block_4(x3)
        x5 = self.down_block_5(x4)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x4], dim=1)
        x5 = self.up_block_4(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_block_3(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x2], dim=1)
        x5 = self.up_block_2(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.up_block_1a(x5)

        output = self.outc(x5)
        return output




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
            nn.Upsample(scale_factor=4.0, mode='nearest')
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
        #print('x5', x5.shape)

        # 8x mix
        x = self.fusion_s1(x1) + self.fusion_s2(x2) + self.fusion_s3(x3) + self.fusion_s5(x5)
        x = self.head(x)
        return x


class DCEPredictor(nn.Module):
    """
    predictor of DCE representation and DensePose 15 coarse segmentation (14 parts / background)
    """
    def __init__(self, dim_in, dce_chan):
        super(DCEPredictor, self).__init__()
        self.dce_chan = dce_chan
        self.decode = BasicBlock(
            dim_in,
            dce_chan,
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
        #p_out, dce_out = torch.split(x, [self.p_chan, self.dce_chan], dim=1)
        dce_out = F.normalize(x, 1)
        return dce_out


class DCEModel(nn.Module):
    def __init__(self, dce_chan, backbone='effunet'):
        super(DCEModel, self).__init__()
        if backbone == 'effunet':
            self.backbone = EffUNet()
            print(sum(p.numel() for p in self.backbone.parameters()))
        else:
            self.backbone = DarkNet()
        self.DCEPredictor = DCEPredictor(dim_in=256, dce_chan=dce_chan)
        print(sum(p.numel() for p in self.DCEPredictor.parameters()))
        self.loss = DenseCorrepondenceEmbeddingLoss(embedding_dim=dce_chan)
        print(sum(p.numel() for p in self.loss.parameters()))
    def forward(self, img, dp_masks_gt=None, dp_x=None, dp_y=None, dp_I=None, dp_U=None, dp_V=None,
                mode='loss'):
        #print(img.shape)
        x = self.backbone(img)
        #dp_masks_pred, 
        dce_pred = self.DCEPredictor(x)
        if mode == 'feat':
            return dce_pred #dp_masks_pred, dce_pred
        elif mode == 'loss':
            return self.loss(
                #dp_masks_pred, 
                dp_masks_gt,
                dce_pred, dp_x, dp_y, dp_I, dp_U, dp_V, img
            )
        elif mode == 'eval':
            self.loss(
                #dp_masks_pred, 
                dp_masks_gt,
                dce_pred, dp_x, dp_y, dp_I, dp_U, dp_V, img, evaluate=True
            )

@register_model
def dce_effunet64(pretrained=True, **kwargs):
    model = DCEModel(dce_chan=64, backbone='effunet')
    return model

@register_model
def dce_darknet19(pretrain=True, backbone='darknet', **kwargs):
    model = DCEModel(dce_chan=4, backbone='darknet')
    return model

@register_model
def dce_darknet19_binary(pretrained=True, backbone='darknet', **kwargs):
    model = DCEModel(dce_chan=32, backbone='darknet')
    return model


@register_model
def dce_darknet19_binary2(pretrained=True, **kwargs):
    model = DCEModel(dce_chan=32, backbone='darknet')
    return model


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from thop import profile
    import timm

    net = timm.create_model('dce_effunet64').cuda()
    net.eval() 

    dummy_img = torch.randn(1, 3, 256, 128).cuda()

    class WrappedDCEModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, img):
           
            return self.model(img, mode='feat')

    wrapped_net = WrappedDCEModel(net)

    flops, params = profile(wrapped_net, inputs=(dummy_img,), verbose=False)

    gflops = flops / 1e9  
    mparams = params / 1e6  

    print("=" * 50)
    print(f"model size: {dummy_img.shape}")
    print(f"total params: {mparams:.2f} M")
    print(f"total computational cost: {gflops:.2f} GFLOPs")
    print("=" * 50)
