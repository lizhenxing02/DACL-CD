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

class ConvNeXt_CD(nn.Module):
    """
    ConvNeXt基底的变化检测模型。
    
    参数:
    - num_classes: 模型需要预测的类别数。
    - pretrained: 是否使用预训练的ResNet50模型作为编码器部分。
    """
    def __init__(self, num_classes, pretrained=None):
        super(ConvNeXt_CD, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # 编码器使用ConvNeXt编码器
        self.encoder = Encoder(pretrained=pretrained)
        upscale = 32
        self.decoder = ConvNeXtDecoder(upscale=upscale,  num_classes=num_classes)

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
        # 修改 ConvNeXt
        # x是输入的两时相图像元组：x[0] = A图，x[1] = B图（与原逻辑一致）
        # 3. 解构Encoder的4个输出特征（必须按顺序：x, diff1, diff2, diff3）
        encoder_feat, diff1, diff2, diff3 = self.encoder(x[0], x[1])
        if return_features:  # 需要返回预测结果和特征
            # 4. 显式传递4个特征给Decoder（而非元组）
            pred = self.decoder(encoder_feat, diff1, diff2, diff3)
            # 返回预测结果 + Encoder的特征（可根据需求选择返回哪些特征，这里保留全部）
            return pred, (encoder_feat, diff1, diff2, diff3)
        else:  # 仅返回预测结果
            return self.decoder(encoder_feat, diff1, diff2, diff3)

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




