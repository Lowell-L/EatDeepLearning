import torch
import torch.nn as nn
from typing import Optional, Callable, Type, Union, List
from torchsummary import summary


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """
    封装 3×3 卷积操作。
    kernel=3，stride=1，padding=1 保证卷积之后 feature map 的尺寸不变
    当 stride=2 时，卷积之后 feature map 的尺寸减半
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False
    )



def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """
    封装 1×1 卷积操作
    kernel=1，stride=1，padding=0 保证卷积之后 feature map 的尺寸不变
    Basicblock 和 Bottleneck 中的 conv1x1 的 stride 均为 1，其作用仅为改变通道数
    downsample 中也使用了 conv1x1，这里 conv1x1 的作用是改变维度实现 +identity。但是这里变化的不仅仅是通道数，可能还有 feature map 尺寸的变化，因此 stride 可能不为 1
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        bias=False
    )


class BasicBlock(nn.Module):
    """
    定义 BasicBlock 模块
    """
    expansion = 1   # （类变量记录）BasicBlock 使通道数不变

    def __init__(
        self,
        in_channels: int,                           # 进入BasicBlock之前 x 的通道数
        out_channels: int,                          # 经BasicBlock处理之后 x 的通道数
        stride: int = 1,                            # block 中涉及卷积的 stride，默认为 1，不改变feature map的尺寸大小
        downsample: Optional[nn.Module] = None,     # 下采样操作，实现维度转换，以进行 out + identity
    ) -> None:
        super(BasicBlock, self).__init__()        

        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)     # 当 conv1 的 stride!=1 时起下采样作用
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)                                                           # inplace=True表示直接对原对象进行修改
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)                   # 通道数不变
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    定义 Bottleneck 模块
    """
    expansion = 4   # （类变量记录）Bottleneck 使通道数变为原来的 4 倍
    
    def __init__(
        self,
        in_channels,                    # 进入Bottleneck之前 x 的通道数            
        mid_channels,                   # x 进入 Bottleneck，首先在conv1减少通道数，in_channels -> mid_channels；然后conv2不涉及通道数的改变，只会通过stride改变feature map的尺寸；conv3增加通道数，mid_channels -> expansion * mid_channels
        stride=1,
        downsample=None,
    ) -> None:
        super().__init__()

        self.conv1 = conv1x1(in_channels, mid_channels)                     # 把通道数变小: in_channels -> mid_channels
        self.bn1   = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride)            # 不改变通道数，当 stride!=1 时会改变 feature map 的尺寸
        self.bn2   = nn.BatchNorm2d(mid_channels)
        self.conv3 = conv1x1(mid_channels, mid_channels * self.expansion)   # 把通道数变大: mid_channels -> expansion * mid_channels
        self.bn3   = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],         # 选择的基础模块，BasicBlock或者Bottleneck
        layers: List[int],                                  # 每个layer中block的个数
        num_classes: int = 1000,                            # 分类任务 类别个数
    ) -> None:
        super().__init__()

        self.in_channels = 64                               # Resnet 的基准通道数，后续的通道数都是 64 的倍数
        
        self.conv1   = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)  # [3, 224, 224] -> [64, 112, 112]
        self.bn1     = nn.BatchNorm2d(self.in_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)             # [64, 112, 112] -> [64, 56, 56]
        
        # layer[x] 表示此layer中block的堆叠个数；stride=1时特征图尺寸不变，stride=2时特征图尺寸减半
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)              # [64, 56, 56] -> [64, 56, 56] -> [256, 56, 56]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)             # [256, 56, 56] -> [128, 56, 56] -> [128, 28, 28] -> [512, 28, 28]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)             # [512, 28, 28] -> [256, 28, 28] -> [256, 14, 14] -> [1024, 14, 14]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)             # [1024, 14, 14] -> [512, 14, 14] -> [512, 7, 7] -> [2048, 7, 7]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                                 # [2048, 7, 7] -> [2048, 1, 1]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv1_channels,         # 表示经过block的conv1处理后feature map的通道数
        blocks,                 # 表示本layer中block的个数
        stride = 1,
    ):
        downsample = None
        if stride != 1 or self.in_channels != conv1_channels * block.expansion:
            # stride != 1 意味着 Basicblock 的 conv1 和 Bottleneck 的 conv2 起下采样的作用，使 feature map 尺寸缩小
            # self.inplanes 相当于这一层layer的输入x
            # planes 相当于这一层layer第一层的通道数
            # block.expansion 是这个 block 的通道扩大倍数
            # x 进入一个block有两条路径，一条是经过block，其通道数变为 planes * block.expansion
            # 另一条路是 identity，通道数不变，依旧是 self.inplanes
            # 因此，当两者不相等时，需要进行 downsample 以实现维度转换
            downsample = nn.Sequential(
                conv1x1(self.in_channels, conv1_channels * block.expansion, stride),
                nn.BatchNorm2d(conv1_channels * block.expansion)
            )
        
        layer = []
        layer.append(
            block(
                self.in_channels, conv1_channels, stride=stride, downsample=downsample   # 通道数发生变化，feature map的尺寸也发生变化
            )
        )

        self.in_channels = conv1_channels * block.expansion                         # 经过layer的第一个block，通道数变为 conv1_channels的expansion倍
        for _ in range(1, blocks):
            layer.append(
                block(self.in_channels, conv1_channels)
            )
        
        return nn.Sequential(*layer)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
resnet101 = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])
summary(resnet101, (3, 224, 224))