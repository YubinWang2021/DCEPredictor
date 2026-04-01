import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from loss.dce_loss import DenseCorrepondenceEmbeddingLoss
import open_clip

class DINOv2VisualWrapper(nn.Module):
    def __init__(self, dinov2_model):
        super().__init__()
        self.dinov2 = dinov2_model
        self.patch_size = 14  
    def forward(self, x):
        B, C, H, W = x.shape
        new_h = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        if H != new_h or W != new_w:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

        with torch.no_grad():
            out = self.dinov2.forward_features(x)
        patch_tokens = out['x_norm_patchtokens']  

        hp = new_h // self.patch_size
        wp = new_w // self.patch_size
        feat = patch_tokens.permute(0, 2, 1).reshape(B, -1, hp, wp)
        return feat

class CLIPVisualWrapper(nn.Module):
    def __init__(self, clip_visual):
        super().__init__()
        self.visual = clip_visual
        self.patch_size = clip_visual.patch_size[0]  

    def forward(self, x):
        B, C, H, W = x.shape
        hp = H // self.patch_size
        wp = W // self.patch_size

        x = self.visual.conv1(x)
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)

        cls = self.visual.class_embedding.unsqueeze(0).repeat(B, 1, 1)
        x = torch.cat([cls, x], dim=1)

        pos = self.visual.positional_embedding.unsqueeze(0)
        cls_pos = pos[:, :1]
        pat_pos = pos[:, 1:]
        orig_s = int(pat_pos.shape[1] ** 0.5)
        pat_pos = pat_pos.reshape(1, orig_s, orig_s, -1).permute(0, 3, 1, 2)
        pat_pos = F.interpolate(pat_pos, size=(hp, wp), mode='bilinear', align_corners=False)
        pat_pos = pat_pos.permute(0, 2, 3, 1).reshape(1, hp*wp, -1)
        pos = torch.cat([cls_pos, pat_pos], dim=1)

        x = x + pos
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)

        feat = x[:, 1:].permute(0, 2, 1).reshape(B, -1, hp, wp)
        return feat


class DINOv2Backbone(nn.Module):
    def __init__(self, out_dim=256, target_size=(32, 16)):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        for p in self.dinov2.parameters():
            p.requires_grad = False
        
        self.wrapper = DINOv2VisualWrapper(self.dinov2)
        self.target_size = target_size

        self.adapter = nn.Sequential(
            nn.Conv2d(384, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.wrapper(x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        x = self.adapter(x)
        return x
    
class CLIPVisualWrapper(nn.Module):
    def __init__(self, clip_visual):
        super().__init__()
        self.visual = clip_visual
        if hasattr(clip_visual, 'patch_size'):
            self.patch_size = clip_visual.patch_size if isinstance(clip_visual.patch_size, int) else clip_visual.patch_size[0]
        elif hasattr(clip_visual, 'conv1'):
            self.patch_size = clip_visual.conv1.kernel_size[0]
        else:
            self.patch_size = 32

    def forward(self, x):
        B, C, H, W = x.shape
        hp = H // self.patch_size
        wp = W // self.patch_size

        if hasattr(self.visual, 'conv1'):
            x = self.visual.conv1(x)  # [B, C, H/P, W/P]
        elif hasattr(self.visual, 'patch_embed'):
            x = self.visual.patch_embed(x)
            hp, wp = x.shape[2], x.shape[3]
        else:
            raise ValueError("OpenCLIP model has no conv1 or patch_embed")

        x = x.flatten(2).transpose(1, 2)  # [B, N, C]

        cls_token = None
        if hasattr(self.visual, 'class_embedding'):
            cls_token = self.visual.class_embedding.unsqueeze(0).repeat(B, 1, 1)
        elif hasattr(self.visual, 'cls_token'):
            cls_token = self.visual.cls_token.repeat(B, 1, 1)
        
        if cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)  # [B, N+1, C]

        pos_emb = None
        if hasattr(self.visual, 'positional_embedding'):
            pos_emb = self.visual.positional_embedding.unsqueeze(0)
        elif hasattr(self.visual, 'pos_embed'):
            pos_emb = self.visual.pos_embed

        if pos_emb is not None and cls_token is not None:
            cls_pos = pos_emb[:, :1, :]
            patch_pos = pos_emb[:, 1:, :]
            
            orig_n = patch_pos.shape[1]
            orig_s = int(orig_n ** 0.5)
            patch_pos = patch_pos.reshape(1, orig_s, orig_s, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(hp, wp), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
            
            new_pos = torch.cat([cls_pos, patch_pos], dim=1)
            x = x + new_pos

        if hasattr(self.visual, 'ln_pre'):
            x = self.visual.ln_pre(x)
        
        if hasattr(self.visual, 'transformer'):
            x = x.permute(1, 0, 2)  # [N+1, B, C]
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # [B, N+1, C]
        
        if hasattr(self.visual, 'ln_post'):
            x = self.visual.ln_post(x)

        if cls_token is not None:
            patch_feat = x[:, 1:, :]
        else:
            patch_feat = x
            
        feat = patch_feat.permute(0, 2, 1).reshape(B, -1, hp, wp)
        return feat
    
class CLIPBackbone(nn.Module):
    def __init__(self, out_dim=256, target_size=(32, 16)):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.visual = self.clip_model.visual
        
        for p in self.visual.parameters():
            p.requires_grad = False

        self.wrapper = CLIPVisualWrapper(self.visual)
        self.target_size = target_size


        self.adapter = nn.Sequential(
            nn.Conv2d(768, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.wrapper(x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        x = self.adapter(x)
        return x


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
        print(feat.shape)
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
        elif backbone == 'unet':
            self.backbone = UNet()
        elif backbone == 'dinov2':       
            self.backbone = DINOv2Backbone()
        elif backbone == 'clip':         
            self.backbone = CLIPBackbone()
        else:
            self.backbone = DarkNet()
        self.DCEPredictor = DCEPredictor(dim_in=256, dce_chan=dce_chan)
        self.loss = DenseCorrepondenceEmbeddingLoss(embedding_dim=dce_chan)

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
def dce_dinov2(pretrained=True, **kwargs):
    model = DCEModel(dce_chan=64, backbone='dinov2')
    return model

@register_model
def dce_clip64(pretrained=True, **kwargs):
    model = DCEModel(dce_chan=64, backbone='clip')
    return model

@register_model
def dce_unet64(pretrained=True, **kwargs):
    model = DCEModel(dce_chan=64, backbone='unet')
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



