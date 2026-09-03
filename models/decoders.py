import math , time
import torch
import torch.nn.functional as F
from torch import nn
# from utils.helpers import initialize_weights
from itertools import chain
import contextlib
import random
import numpy as np
import cv2
from torch.distributions.uniform import Uniform


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

# 像素堆叠
class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


# def upsample(in_channels, out_channels, upscale, kernel_size=3):
#     # A series of x 2 upsamling until we get to the upscale we want
#     layers = []
#     conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#     nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
#     layers.append(conv1x1)
#     for i in range(int(math.log(upscale, 2))):
#         layers.append(PixelShuffle(out_channels, scale=2))
#     return nn.Sequential(*layers)

def upsample(in_channels, out_channels, upscale, kernel_size=3):
    layers = []

    # Middle channels 中间特征图的通道数
    mid_channels = 32

    #First conv layer to reduce number of channels 第一个卷积层用于减少通道数量
    diff_conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False)
    nn.init.kaiming_normal_(diff_conv1x1.weight.data, nonlinearity='relu')
    layers.append(diff_conv1x1)

    #ReLU
    diff_relu = nn.ReLU()
    layers.append(diff_relu)

    #Upsampling to original size  上采样至原始尺寸
    # up      = nn.Upsample(scale_factor=upscale, mode='bilinear')
    up = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=True)
    layers.append(up)

    #Classification layer  分类层
    conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)

    return nn.Sequential(*layers)

# 主解码器
class MainDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(MainDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(x)
        return x

# 添加适配ConvNeXt-tiny解码器
# 辅助模块：卷积+BN+ReLU（提升特征表达能力，适配ConvNeXt的特征分布）
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),  # 补充BN层，稳定特征分布
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

# 重写MainDecoder：多尺度特征融合+逐步上采样
class ConvNeXtDecoder(nn.Module):
    def __init__(self, upscale, num_classes):
        super(ConvNeXtDecoder, self).__init__()
        self.upscale = upscale  # 总上采样倍数（应为32，与Encoder下采样匹配）
        # ------------ 第一层：处理Encoder最终特征x（32x下采样，768通道）------------
        self.conv4 = ConvBNReLU(768, 512, 3, 1)  # 768→512（平缓降维）
        # 上采样：32x → 16x（与diff3尺度匹配）
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # ------------ 第二层：融合diff3（16x下采样，384通道）------------
        self.conv3 = ConvBNReLU(512 + 384, 256, 3, 1)  # 512（上采样后）+ 384（diff3）= 832 → 256
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 16x→8x
        # ------------ 第三层：融合diff2（8x下采样，192通道）------------
        self.conv2 = ConvBNReLU(256 + 192, 128, 3, 1)  # 256 + 192 = 448 → 128
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 8x→4x
        # ------------ 第四层：融合diff1（4x下采样，96通道）------------
        self.conv1 = ConvBNReLU(128 + 96, 64, 3, 1)  # 128 + 96 = 224 → 64
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 4x→原图（H/W）
        # ------------ 最终分类层 ------------
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.final_conv.weight.data, nonlinearity='relu')

    def forward(self, x, diff1, diff2, diff3):
        # 第一步：处理32x特征 → 16x，融合diff3
        x4 = self.conv4(x)          # (N,512,H/32,W/32)
        x4_up = self.up4(x4)        # (N,512,H/16,W/16)
        x3 = torch.cat([x4_up, diff3], dim=1)  # 融合diff3：(N,512+384=832,H/16,W/16)
        x3 = self.conv3(x3)         # (N,256,H/16,W/16)
        # 第二步：16x → 8x，融合diff2
        x3_up = self.up3(x3)        # (N,256,H/8,W/8)
        x2 = torch.cat([x3_up, diff2], dim=1)  # 融合diff2：(N,256+192=448,H/8,W/8)
        x2 = self.conv2(x2)         # (N,128,H/8,W/8)
        # 第三步：8x → 4x，融合diff1
        x2_up = self.up2(x2)        # (N,128,H/4,W/4)
        x1 = torch.cat([x2_up, diff1], dim=1)  # 融合diff1：(N,128+96=224,H/4,W/4)
        x1 = self.conv1(x1)         # (N,64,H/4,W/4)
        # 第四步：4x → 原图尺寸，输出分类结果
        x_final = self.up1(x1)      # (N,64,H,W)
        out = self.final_conv(x_final)  # (N,num_classes,H,W)
        
        return out

class DropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _, pertub=True):
        if pertub:
            x = self.upsample(self.dropout(x))
        else:
            x = self.upsample(x)
        return x

# 特征丢弃解码器
class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _, pertub=True):
        if pertub:
            x = self.feature_dropout(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x

# 特征噪声解码器
class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _, pertub=True):
        if pertub:
            x = self.feature_based_noise(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x



def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv

# VAT解码器
class VATDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _, pertub=True):
        if pertub:
            r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
            x = self.upsample(x + r_adv)
        else:
            x = self.upsample(x)
        return x



def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6)/10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        # maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest', align_corners=None)

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try: # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except: # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w-min_w, max_h-min_h
            rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
            rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
            h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
            w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    # maskcut = F.interpolate(maskcut, size=resize, mode='nearest')
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest', align_corners=None)

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)

# 剪切解码器
class CutOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale 
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None, pertub=True):
        if pertub:
            maskcut = guided_cutout(pred, upscale=self.upscale, erase=self.erase, resize=(x.size(2), x.size(3)))
            x = x * maskcut
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x


def guided_masking(x, output, upscale, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
    
    # masks_context = F.interpolate(masks_context, size=resize, mode='nearest')
    masks_context = F.interpolate(masks_context, size=resize, mode='nearest', align_corners=None)

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects

# 上下文掩码解码器
class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None, pertub=True):
        if pertub:
            x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          upscale=self.upscale, return_msk_context=True)
            x_masked_context = self.upsample(x_masked_context)
        else:
            x_masked_context = self.upsample(x)
        return x_masked_context

# 对象掩码解码器
class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None, pertub=True):
        if pertub:
            x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
            x_masked_obj = self.upsample(x_masked_obj)
        else:
            x_masked_obj = self.upsample(x)
        return x_masked_obj

