from models.backbones.resnet_backbone import ResNetBackbone
from models.backbones.efficientnet_backbone import EfficientNet
from efficientnet_pytorch import EfficientNet
# from utils.helpers import initialize_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# from timm.models import convnext_tiny
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

resnet50 = {
    "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",
}
resnet18 = {
    "path": "models/backbones/pretrained/resnet18-5c106cde.pth",
}

#对应论文中MGSAM模块中的FDM（特征差异模块），用于构建具有多个 3x3 卷积、批归一化、ReLU 激活和 Dropout 的网络层结构
def Conv3x3ReLUBNs(in_channels,
                   inner_channels,
                   num_convs): #输入通道数 in_channels、内部通道数 inner_channels 和卷积层的数量 num_convs

    layers = [nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(True),
        nn.Dropout()
    )]
    layers += [nn.Sequential(
        nn.Conv2d(inner_channels, inner_channels, 3, 1, 1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(True),
        nn.Dropout()    
    ) for _ in range(num_convs - 1)] #根据 num_convs 的数量，添加额外的相同结构的序列模块（除第一个外）

    return nn.Sequential(*layers) #将所有的序列模块组合成一个顺序模块并返回

#特征差分加权空间注意力模块（FDSAM）
class metric_attention(nn.Module): 
    def __init__(self, in_channels=256, inner_channels=256, num_convs=2):
        super(metric_attention, self).__init__()
        self.compare = Conv3x3ReLUBNs(in_channels=in_channels, inner_channels=inner_channels, num_convs=num_convs) # FDM

        #对应论文中FDSAM模块的特征转换层（FTL）
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1), 
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels//4, kernel_size=1, stride=1), 
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
        )

        self.sigmoid = nn.Sigmoid() #用于生成注意力权重

    def forward(self, x1, x2):
        #计算 x1 和 x2 的差的绝对值，并通过 compare 模块进行处理
        change_coarse = self.compare(torch.abs(x1-x2)) 
        #对 x1 和 x2 进行压缩操作、归一化，并计算范数差，得到度量值
        metric = torch.norm(F.normalize(self.squeeze(x1), dim=1, eps=1e-6)-
                            F.normalize(self.squeeze(x2), dim=1, eps=1e-6), 
                            dim=1, keepdim=True)
        optimized_diff = change_coarse * self.sigmoid(metric)  # 空间注意力加权后的差异特征
        return optimized_diff

class _PSPModule(nn.Module):
    """
    Pyramid Scene Parsing (PSP) 模块。
    该模块通过不同大小的池化窗口对输入特征图进行下采样，以捕获不同尺度的特征，
    然后通过上采样将这些特征图融合在一起，以获得更丰富的上下文信息。
    参数:
        in_channels (int): 输入特征图的通道数。
        bin_sizes (list of int): 池化窗口的大小列表，每个大小对应一个尺度的特征。
    """
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()
        # 根据bin_sizes的长度，确定每个尺度特征图的通道数
        out_channels = in_channels // len(bin_sizes)
        # 创建不同尺度的特征提取模块
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        # 创建一个瓶颈卷积层，用于融合原始特征和不同尺度的特征
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    #创建特定尺度的特征提取模块
    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        """
        前向传播。
        
        参数:
            features (Tensor): 输入的特征图。
        返回:
            output (Tensor): 融合不同尺度特征后的输出特征图。
        """
        h, w = features.size()[2], features.size()[3]
        # 初始化pyramids列表，首先包含原始特征图
        pyramids = [features]
        # 将每个尺度的特征图通过上采样后添加到pyramids列表中
        # pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
        #                                 align_corners=False) for stage in self.stages])
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        # 将原始特征和不同尺度的特征图在通道维度上进行拼接，并通过瓶颈卷积层进行融合
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output



class Encoder(nn.Module):
    """
    Encoder类继承自nn.Module，用于编码输入图像的特征。
    参数:
    - pretrained (bool): 是否使用预训练的ResNet模型。
    属性:
    - model: ResNet50_CD模型实例。
    - out_channels (int): 编码器的输出通道数。
    - fdsam (metric_attention): 特征差分加权空间注意力模块。
    - psp (_PSPModule): 金字塔场景解析模块，用于增强全局上下文理解。
    """
    def __init__(self, pretrained):
        super(Encoder, self).__init__()

        # # 检查是否需要下载预训练的ResNet模型
        # if pretrained and not os.path.isfile(resnet50["path"]):
        #     print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
        #     # os.system('sh models/backbones/get_pretrained_model.sh')
        # # 调用models中resnet_backbone.py中的ResNetBackbone来导入resnet
        # model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained)
        # # 构建基础模型，包括前置处理、最大池化和四个残差块
        # self.base = nn.Sequential(
        #     nn.Sequential(model.prefix, model.maxpool),
        #     model.layer1,
        #     model.layer2,
        #     model.layer3,
        #     model.layer4
        # )
        # # 初始化金字塔场景解析模块，用于特征图的多尺度上下文聚合
        # self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])
        # # 初始化特征差分加权空间注意力模块
        # self.fdsam = metric_attention(in_channels=2048, inner_channels=2048, num_convs=2)

        # # 改为ResNet18
        # if pretrained and not os.path.isfile(resnet18["path"]):
        #     print("Downloading pretrained resnet (source : https://download.pytorch.org/models/resnet18-5c106cde.pth)")
        #     # os.system('sh models/backbones/get_pretrained_model.sh')
        # # 调用models中resnet_backbone.py中的ResNetBackbone来导入resnet
        # model = ResNetBackbone(backbone='resnet18_dilated8', pretrained=pretrained)
        # # 构建基础模型，包括前置处理、最大池化和四个残差块
        # self.base = nn.Sequential(
        #     nn.Sequential(model.prefix, model.maxpool),
        #     model.layer1,
        #     model.layer2,
        #     model.layer3,
        #     model.layer4
        # )
        # # 初始化金字塔场景解析模块，用于特征图的多尺度上下文聚合
        # self.psp = _PSPModule(512, bin_sizes=[1, 2, 3, 6])
        # # 初始化特征差分加权空间注意力模块
        # self.fdsam = metric_attention(in_channels=512, inner_channels=512, num_convs=2)


        # # 第三次修改 改空间注意力模块
        # if pretrained:
        #     # 加载预训练的 EfficientNet 权重
        #     self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # else:
        #     # 不加载预训练权重，从头开始训练
        #     self.model = EfficientNet.from_name('efficientnet-b0')  # 使用 from_name 创建未预训练的模型
        # # 移除原始分类头
        # self.model._fc = nn.Identity()
        # # 获取编码器的输出通道数
        # self.out_channels = self.model._conv_head.out_channels
        # # 初始化特征差分加权空间注意力模块
        # self.fdsam = metric_attention(in_channels=self.out_channels, inner_channels=self.out_channels, num_convs=2)
        # # 初始化金字塔场景解析模块，用于特征图的多尺度上下文聚合
        # self.psp = _PSPModule(self.out_channels, bin_sizes=[1, 2, 3, 6])


        # 替换为ConvNeXt
        if pretrained:
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            model = convnext_tiny(weights=None)
        self.stem = model.features[0]  # 对应初始层：输出96（self.base[0]）
        self.stage1 = model.features[1]  # 第一个stage：输出96（self.base[1]）
        self.stage2 = nn.Sequential(model.features[2], model.features[3])  # 第二个stage：输出192（self.base[2]）
        self.stage3 = nn.Sequential(model.features[4], model.features[5])  # 第三个stage：输出384（self.base[3]）
        self.stage4 = nn.Sequential(model.features[6], model.features[7])  # 第四个stage：输出768（self.base[4]）

        # self.base = nn.Sequential(
        #     self.stem,  # ConvNeXt的stem层（替代ResNet的prefix+maxpool）
        #     self.stage1,  # 第一个特征阶段
        #     self.stage2,  # 第二个特征阶段
        #     self.stage3,  # 第三个特征阶段
        #     self.stage4   # 第四个特征阶段（最终输出通道768）
        # )
        self.fdsam = metric_attention(in_channels=768, inner_channels=768, num_convs=2)
        # self.psp = _PSPModule(768, bin_sizes=[1, 2, 3, 6])

    def forward(self, A, B):
        """
        前向传播函数，处理两个输入并计算它们的特征差异。
        参数:
        - A (Tensor): 第一个输入图像张量。
        - B (Tensor): 第二个输入图像张量。
        返回:
        - x (Tensor): 编码器的输出————特征差异张量。
        """
        # a = self.base(A) # 获取特征图
        # b = self.base(B)
        # optimized_diff = self.fdsam(a, b)  # shape: (N, self.out_channels, H, W)
        # x = self.psp(optimized_diff)  # shape: (N, out_channels_psp, H, W)，其中out_channels_psp = self.out_channels // 4
        # return x

        # # 第三次修改 改空间注意力模块 EfficientNetB0
        # a = self.model.extract_features(A) # 获取特征图
        # b = self.model.extract_features(B)
        # # 用FDSAM优化差异特征（替代原有的简单绝对差异）
        # optimized_diff = self.fdsam(a, b)  # shape: (N, self.out_channels, H, W)
        # # 用PSP增强多尺度上下文信息
        # x = self.psp(optimized_diff)  # shape: (N, out_channels_psp, H, W)，其中out_channels_psp = self.out_channels // 4
        # return x

        # 从ConvNeXt提取特征（输入为A和B两个时相的图像）
        # A图分步提取各stage特征（保留中间特征）
        a_stem = self.stem(A)          # A图：(N, 96, H/4, W/4)
        a1 = self.stage1(a_stem)       # A图stage1：(N, 96, H/4, W/4)
        a2 = self.stage2(a1)           # A图stage2：(N, 192, H/8, W/8)
        a3 = self.stage3(a2)           # A图stage3：(N, 384, H/16, W/16)
        a4 = self.stage4(a3)           # A图stage4：(N, 768, H/32, W/32)
        # B图
        b_stem = self.stem(B)          # B图：(N, 96, H/4, W/4)
        b1 = self.stage1(b_stem)       # B图stage1：(N, 96, H/4, W/4)
        b2 = self.stage2(b1)           # B图stage2：(N, 192, H/8, W/8)
        b3 = self.stage3(b2)           # B图stage3：(N, 384, H/16, W/16)
        b4 = self.stage4(b3)           # B图stage4：(N, 768, H/32, W/32)
        # FDSAM（差异优化）和PSP（多尺度融合）
        optimized_diff = self.fdsam(a4, b4)  # 基于stage4特征的差异优化：(N,768,H/32,W/32)
        # x = self.psp(optimized_diff)        # PSP融合后：(N,768,H/32,W/32)
        x = optimized_diff               # 取消PSP，直接使用优化后的差异特征
        # 返回：最终特征x + 各stage的A/B差异特征（用于Decoder融合）
        # 计算各stage的A/B差异（变化检测核心：捕捉两时相差异）
        diff1 = torch.abs(a1 - b1)  # stage1差异：(N,96,H/4,W/4)（细节差异）
        diff2 = torch.abs(a2 - b2)  # stage2差异：(N,192,H/8,W/8)（中低语义差异）
        diff3 = torch.abs(a3 - b3)  # stage3差异：(N,384,H/16,W/16)（中高语义差异）
        
        return x, diff1, diff2, diff3  # x：最终高维特征；diff1-diff3：多尺度差异特征

    # 第二次修改
    def get_out_channels(self):
        """
        获取编码器的输出通道数。
        """
        return self.out_channels

    #获取基础模型（backbone）的参数
    def get_backbone_params(self):
        return self.base.parameters()

    #获取PSP模块的参数
    def get_module_params(self):
        return self.psp.parameters()
