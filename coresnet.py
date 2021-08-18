import torch
import torch.nn as nn
import os
from div.download_from_url import download_from_url

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pretrained')

__all__ = ['CoResNet', 'coresnet50', 'coresnet101', 'coresnet152']


model_urls = {
    'coresnet50': 'https://drive.google.com/uc?export=download&id=1fnANwWhU6SjRpPLSHIji-nLNhdVdtBSs',
    'coresnet101': 'https://drive.google.com/uc?export=download&id=10-oJYZtPrlnM4H9aKxvI-Osjhjm5B1WL',
    'coresnet152': 'https://drive.google.com/uc?export=download&id=1dzo1gd7l6_T57wWcfyxSddGlec1foClD',
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CoConv3x3_2d(nn.Module):
    """CoConv3x3_2d with padding (general case). Applies a 2D CoConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        coconv_dilations (list): The dilation of the kernel for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        groups (int): Number of groups to split the input channels. Default: 1

    Example::

        >>> # CoConv with two pyramid levels, dilations: 1, 2
        >>> m = CoConv3x3_2d(in_channels=64, out_channels=[32, 32], coconv_dilations=[1, 2])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # CoConv with three pyramid levels, kernels: 1, 2, 3
        >>> m = CoConv3x3_2d(in_channels=64, out_channels=[32, 16, 16], coconv_dilations=[1, 2, 3])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, coconv_dilations, stride=1, bias=False, groups=1):
        super(CoConv3x3_2d, self).__init__()
        assert len(out_channels) == len(coconv_dilations)
        kernel_size = 3

        self.coconv_levels = [None] * len(coconv_dilations)
        for i in range(len(coconv_dilations)):

            self.coconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=kernel_size,
                                              stride=stride, padding=kernel_size // 2 + (coconv_dilations[i] - 1),
                                              dilation=coconv_dilations[i], bias=bias, groups=groups)
        self.coconv_levels = nn.ModuleList(self.coconv_levels)

    def forward(self, x):
        out = []
        for level in self.coconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)

class CoConv5(nn.Module):

    def __init__(self, inplans, planes, stride=1, groups=1):
        super(CoConv5, self).__init__()
        self.conv2_1 = conv3x3(inplans, planes//4,
                               stride=stride, padding=1, dilation=1, groups=groups)
        self.conv2_2 = conv3x3(inplans, planes // 4, stride=stride, padding=2, dilation=2, groups=groups)
        self.conv2_3 = conv3x3(inplans, planes // 4, stride=stride, padding=3, dilation=3, groups=groups)
        self.conv2_4 = conv3x3(inplans, planes // 8, stride=stride, padding=4, dilation=4, groups=groups)
        self.conv2_5 = conv3x3(inplans, planes // 8, stride=stride, padding=5, dilation=5, groups=groups)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x), self.conv2_5(x)), dim=1)


class CoConv4(nn.Module):

    def __init__(self, inplans, planes, stride=1, groups=1):
        super(CoConv4, self).__init__()
        self.conv2_1 = conv3x3(inplans, planes // 4, stride=stride, padding=1,
                               dilation=1, groups=groups)
        self.conv2_2 = conv3x3(inplans, planes // 4, stride=stride, padding=2, dilation=2, groups=groups)
        self.conv2_3 = conv3x3(inplans, planes // 4, stride=stride, padding=3, dilation=3, groups=groups)
        self.conv2_4 = conv3x3(inplans, planes // 4, stride=stride, padding=4, dilation=4, groups=groups)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class CoConv3(nn.Module):

    def __init__(self, inplans, planes, stride=1, groups=1):
        super(CoConv3, self).__init__()
        self.conv2_1 = conv3x3(inplans,  planes // 2, stride=stride, padding=1,
                               dilation=1, groups=groups)
        self.conv2_2 = conv3x3(inplans, planes // 4, stride=stride, padding=2, dilation=2, groups=groups)
        self.conv2_3 = conv3x3(inplans, planes // 4, stride=stride, padding=3, dilation=3, groups=groups)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class CoConv2(nn.Module):

    def __init__(self, inplans, planes, stride=1, groups=1):
        super(CoConv2, self).__init__()
        self.conv2_1 = conv3x3(inplans, planes // 2, stride=stride, padding=1,
                               dilation=1, groups=groups)
        self.conv2_2 = conv3x3(inplans, planes // 2, stride=stride, padding=2, dilation=2, groups=groups)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def create_spatial_conv(inplans, planes, pyramid_levels, stride=1, groups=1):
    if pyramid_levels == 1:
        return conv3x3(inplans, planes, stride=stride, groups=groups)
    elif pyramid_levels == 2:
        return CoConv2(inplans, planes, stride=stride, groups=groups)
    elif pyramid_levels == 3:
        return CoConv3(inplans, planes, stride=stride, groups=groups)
    elif pyramid_levels == 4:
        return CoConv4(inplans, planes, stride=stride, groups=groups)
    elif pyramid_levels == 5:
        return CoConv5(inplans, planes, stride=stride, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, groups=1, pyramid_levels=1,
                 bn_end_stage=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = create_spatial_conv(planes, planes, pyramid_levels=pyramid_levels, stride=stride, groups=groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.bn_end_stage = bn_end_stage

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if not self.bn_end_stage:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.bn_end_stage:
            out = self.bn3(out)

        out = self.relu(out)

        return out


class CoResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None, dropout_prob0=0.0):
        super(CoResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, norm_layer=norm_layer,
                                       pyramid_levels=4, groups_block=1, bn_end_stage=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer,
                                       pyramid_levels=3, groups_block=1, bn_end_stage=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer,
                                       pyramid_levels=2, groups_block=1, bn_end_stage=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer,
                                       pyramid_levels=1, groups_block=1, bn_end_stage=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
            print("Using Dropout with the prob to set to 0 of: ", dropout_prob0)
        else:
            self.dp = None

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None,
                    pyramid_levels=1, groups_block=1, bn_end_stage=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer,
                            pyramid_levels=pyramid_levels, groups=groups_block))
        self.inplanes = planes * block.expansion
        for _ in range(1, (blocks-1)):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                pyramid_levels=pyramid_levels, groups=groups_block))

        layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                            pyramid_levels=pyramid_levels, groups=groups_block, bn_end_stage=bn_end_stage))

        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)

        if self.dp is not None:
            x = self.dp(x)

        x = self.fc(x)

        return x


def coresnet50(pretrained=False, **kwargs):
    """Constructs a CoResNet-50 model.
    """
    model = CoResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['coresnet50'],
                                                           root=default_cache_path)))
    
    return model


def coresnet101(pretrained=False, **kwargs):
    """Constructs a CoResNet-101 model.
    """
    model = CoResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['coresnet101'],
                                                           root=default_cache_path)))
    
    return model


def coresnet152(pretrained=False, **kwargs):
    """Constructs a CoResNet-152 model.
    """
    model = CoResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['coresnet152'],
                                                           root=default_cache_path)))
    
    return model

# CoConv with two pyramid levels, kernels: 3x3, 5x5
m = CoConv3x3_2d(in_channels=64, out_channels=[32, 32, 16], coconv_dilations=[1, 2, 3])
input = torch.randn(4, 64, 56, 56)
output = m(input)
#print(m)
#print(output.shape)
m = coresnet152(pretrained=True)
print(m.state_dict())
