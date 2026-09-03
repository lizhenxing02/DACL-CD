import math
import torch
import torch.nn as nn

# 激活函数需要继承nn.Module，然后进行正向传播
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 3x3 DW卷积  其中'//'是取整运算符，eg: 3//2=1,
# 其中stride默认为1，也就是一般不需要进行尺寸减半操作，而通过padding扩充使得对于kernel_size为3/5均可自动补充
def ConvBNAct(in_channels,out_channels,kernel_size=3, stride=1, groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups),
            nn.BatchNorm2d(out_channels),
            Swish()
        )


# pw卷积含激活函数
def Conv1x1BNAct(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            Swish()
        )


# pw卷积不含激活函数
def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        Swish(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


# 打平操作
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# SE结构
class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        # 这里压缩的倍数的16倍，而在MobileNetv3中压缩的倍数是4倍
        mid_channels = channels // ratio
        # 基于通道的注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


# MobileNetv3中的结构块，所以是MBBlock，需要注意，channels的扩充有因子expansion_factor=6
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expansion_factor = expansion_factor
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            # 首先PW卷积升维，扩充倍数是6倍
            Conv1x1BNAct(in_channels, mid_channels),
            # 然后DW卷积，维度不变，并且通过这一步将尺寸缩放
            ConvBNAct(mid_channels, mid_channels, kernel_size, stride, groups=mid_channels),
            # 添加一个注意力模块
            SEBlock(mid_channels),
            # 最后PW卷积降维操作，变回原来的channels数
            Conv1x1BN(mid_channels, out_channels)
        )

        # 残差相加
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride==1 else out
        return out


# 主网络
class EfficientNet(nn.Module):
    params = {
        'efficientnet_b0': (1.0, 1.0, 224, 0.2),
        'efficientnet_b1': (1.0, 1.1, 240, 0.2),
        'efficientnet_b2': (1.1, 1.2, 260, 0.3),
        'efficientnet_b3': (1.2, 1.4, 300, 0.3),
        'efficientnet_b4': (1.4, 1.8, 380, 0.4),
        'efficientnet_b5': (1.6, 2.2, 456, 0.4),
        'efficientnet_b6': (1.8, 2.6, 528, 0.5),
        'efficientnet_b7': (2.0, 3.1, 600, 0.5),
    }

    def __init__(self, subtype = 'efficientnet_b0', num_classes = 2):
        super(EfficientNet, self).__init__()
        self.width_coeff = self.params[subtype][0]      # 元组的第一个参数: 宽度的扩充因子 width_coefficient
        self.depth_coeff = self.params[subtype][1]      # 元组的第二个参数: 深度的扩充因子 depth_coefficient
        self.dropout_rate = self.params[subtype][3]     # 元组的第四个参数: 随机丢弃比例 最后一个全连接层前的dropout层
        self.depth_div = 8

        # 对于b0层: 输出的channels为32，特征尺寸需要减半处理，所以stride=2
        self.stage1 = ConvBNAct(3, self._calculate_width(32), kernel_size=3, stride=2)
        # 此处stride=1，所以尺寸不变
        self.stage2 = self.make_layer(self._calculate_width(32), self._calculate_width(16), kernel_size=3, stride=1, block=self._calculate_depth(1))
        self.stage3 = self.make_layer(self._calculate_width(16), self._calculate_width(24), kernel_size=3, stride=2, block=self._calculate_depth(2))
        self.stage4 = self.make_layer(self._calculate_width(24), self._calculate_width(40), kernel_size=5, stride=2, block=self._calculate_depth(2))
        self.stage5 = self.make_layer(self._calculate_width(40), self._calculate_width(80), kernel_size=3, stride=2, block=self._calculate_depth(3))
        # 此处stride=1，所以尺寸不变
        self.stage6 = self.make_layer(self._calculate_width(80), self._calculate_width(112), kernel_size=5, stride=1, block=self._calculate_depth(3))
        self.stage7 = self.make_layer(self._calculate_width(112), self._calculate_width(192), kernel_size=5, stride=2, block=self._calculate_depth(4))
        self.stage8 = self.make_layer(self._calculate_width(192), self._calculate_width(320), kernel_size=3, stride=1, block=self._calculate_depth(1))

        # 池化操作
        self.pooling = nn.Sequential(
            Conv1x1BNAct(self._calculate_width(320), self._calculate_width(1280)),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.2),
        )

        # 全连接层
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self._calculate_width(1280), num_classes)
        )

        self.init_weights()

    # 参数初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    # width_coefficient因子取整到离它最近的8的整数倍
    def _calculate_width(self, x):
        # 对于输入的channels，需要乘上宽度的扩展因子
        x *= self.width_coeff
        # 当x偏向8整数倍的下方，通过加上4来判断
        new_x = max(self.depth_div, int(x + self.depth_div / 2) // self.depth_div * self.depth_div)
        if new_x < 0.9 * x:
            new_x += self.depth_div
        return int(new_x)

    # depth_coefficient因子得到小数后向上取整
    def _calculate_depth(self, x):
        return int(math.ceil(x * self.depth_coeff))     # ceil向上取整函数 eg: ceil(4.01)=5

    def make_layer(self, in_places, places, kernel_size, stride, block):
        layers = []
        # 通过块结构的第一层改变尺寸，所以需要传入stride
        layers.append(MBConvBlock(in_places, places, kernel_size, stride))
        # 块结构的其他层不改变channels。注意，此处的in_channels = out_channels = places
        for i in range(1, block):
            layers.append(MBConvBlock(places, places, kernel_size))
        return nn.Sequential(*layers)

    # 以基本结构b0为例:
    # input: torch.Size([1, 3, 224, 224])
    def forward(self, x):
        x = self.stage1(x)      # torch.Size([1, 32, 112, 112])
        x = self.stage2(x)      # torch.Size([1, 16, 112, 112])
        x = self.stage3(x)      # torch.Size([1, 24, 56, 56])
        x = self.stage4(x)      # torch.Size([1, 40, 28, 28])
        x = self.stage5(x)      # torch.Size([1, 80, 14, 14])
        x = self.stage6(x)      # torch.Size([1, 112, 14, 14])
        x = self.stage7(x)      # torch.Size([1, 192, 7, 7])
        x = self.stage8(x)      # torch.Size([1, 320, 7, 7])
        x = self.pooling(x)     # torch.Size([1, 1280, 1, 1])
        x = self.fc(x)          # torch.Size([1, 5])
        return x

