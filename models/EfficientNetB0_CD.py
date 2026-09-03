import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
# from base import BaseModel
# from utils.helpers import set_trainable
# from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
# from utils.losses import CE_loss

class EfficientNetB0_CD(nn.Module):
    """
    EfficientNet基底的变化检测模型。
    
    参数:
    - num_classes: 模型需要预测的类别数。
    - pretrained: 是否使用预训练的ResNet50模型作为编码器部分。
    """
    def __init__(self, num_classes, pretrained=None):
        super(EfficientNetB0_CD, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # # 创建编码器，如果指定预训练，则使用预训练的ResNet50模型
        # self.encoder = Encoder(pretrained=pretrained)
        # # 定义模型的放大倍数、输出通道数和解码器的输入通道数
        # upscale = 8
        # num_out_ch = 2048
        # decoder_in_ch = num_out_ch // 4
        # # 创建解码器，负责将编码器的输出转换为最终的类别预测
        # self.decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)

        # # 修改 改空间注意力模块 EfficientNetB0
        # self.encoder = Encoder(pretrained=pretrained)
        # encoder_out_channels = self.encoder.get_out_channels()
        # upscale = 32
        # self.decoder = MainDecoder(upscale, encoder_out_channels//4, num_classes=num_classes)     

        # 编码器使用修改后的ConvNeXt编码器
        self.encoder = Encoder(pretrained=pretrained)
        encoder_out_channels = self.encoder.get_out_channels()
        # 解码器参数适配：ConvNeXt的下采样倍数为32（与EfficientNet相同），无需修改upscale
        upscale = 32
        self.decoder = MainDecoder(upscale, encoder_out_channels//4, num_classes=num_classes)

    def forward(self, x, return_features=False):
        """
        模型的前向传播函数。
        参数:
        - x: 输入数据，包含两个时间点的图像数据。
        - return_features: 是否返回特征图。
        返回:
        - 如果return_features为True，则返回变更预测和特征图。
        - 如果return_features为False，则只返回变更预测。
        """
        if return_features:  # return change predictions and features
            features = self.encoder(x[0], x[1]) # 获取特征图
            return self.decoder(features), features # 返回变化预测和特征图
        else:
            return self.decoder(self.encoder(x[0], x[1])) # 只返回变化预测

    def pretrained_parameters(self):
        """
        获取预训练模型的参数。
        
        返回:
        - 如果使用预训练模型，则返回预训练模型的参数列表。
        - 如果不使用预训练模型，则返回空列表。
        """
        if self.pretrained:
            return list(self.encoder.get_backbone_params()) # 返回预训练模型的参数列表
        else:
            return []

    def new_parameters(self):
        """
        获取新添加的模型参数。
        
        返回:
        - 如果使用预训练模型，则返回除了预训练模型参数外的所有参数列表。
        - 如果不使用预训练模型，则返回所有参数的列表。
        """
        if self.pretrained:
            pretrained_ids = [id(p) for p in self.encoder.get_backbone_params()] # 获取预训练参数的ID
            return [p for p in self.parameters() if id(p) not in pretrained_ids] # 返回新参数列表
        else:
            return list(self.parameters()) # 返回所有参数列表

# if __name__ == "__main__":
#     net = ResNet50_CD(num_classes=2, pretrained=None)
#     A = torch.rand([4,3,256,256])
#     B = torch.rand([4,3,256,256])
#     out = net(A,B)
#     print(out.shape)




