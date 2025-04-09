import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import reduce
from torch.autograd import Variable
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            # print(m.name)
            # if classname.find('pretrained') != -1:
            #     print(classname)
            #     return
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape

    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_shape[0], 48, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
        ))
    scratch.layer2_rn = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_shape[1], 96, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
        ))
    scratch.layer3_rn = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_shape[2], 192, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
        ))
    scratch.layer4_rn = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_shape[3], 384, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
        ))

    return scratch


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=0, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=0, bias=True, groups=self.groups
        )

        self.pad1 = nn.ReflectionPad2d(1)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.pad1(out)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.pad1(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        # align_corners=True:每个像素的在矩阵里的下标i,j被直接视作坐标系里的一个个的坐标点进行计算
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        # "rwightman/gen-efficientnet-pytorch",
        "/media/whealther/diskb/UWICN-master/gen-efficientnet-pytorch-master",
        # "/home/YanHR/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master",
        # '/data1/ALLData/yhr/cycle/pre_trained/rwightman_gen-efficientnet-pytorch_master'
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
        source='local'
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x


class DepthEstimationNet(BaseNetwork):
    def __init__(self, base_channel_nums=48, init_weight=False, use_pretrained=False, min_d=0.3, max_d=10):
        super().__init__()
        super(DepthEstimationNet, self).__init__()
        # 参数设置
        self.MAX_D = max_d
        self.MIN_D = min_d
        backbone = "efficientnet_lite3"
        exportable = True
        align_corners = True
        blocks = {'expand': True}
        features = base_channel_nums

        self.groups = 1
        self.blocks = blocks
        self.backbone = backbone
        self.excpand = False

        # 输入输出通道数初始化
        features1 = features
        features2 = features
        features3 = features
        features4 = features
        # 通道数增加
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        # self.expand = True

        # 传入参数分别为：输入通道数、输出通道数、卷积组数（决定是否采用分组卷积）、输出通道数是否增加
        # 该函数构建了一个4层卷积层
        # self.scratch = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)
        self.scratch = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)

        # 设置网络的激活函数
        self.scratch.activation = nn.ReLU(False)

        # 传入参数分别为：输入通道数，激活函数类型，是否进行上采样，是否进行批归一化，输出通道数是否改变
        # 该函数
        self.scratch.refinenet4 = FeatureFusionBlock_custom(384, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(192, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(96, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(48, self.scratch.activation, deconv=False, bn=False,
                                                            align_corners=align_corners)

        # 图像输出卷积
        # 该函数用于减小图像的通道数，最终输出图像的通道数为1
        self.scratch.output_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=0, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=0),
            self.scratch.activation,
            # the out_channels has been changed from 1 to 3
            nn.Conv2d(12, 1, kernel_size=1, stride=1, padding=0),
        )

        # self.scratch.output_conv = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=0, groups=self.groups),
        #     Interpolate(scale_factor=2, mode="bilinear"),
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=0),
        #     self.scratch.activation,
        #     # the out_channels has been changed from 1 to 3
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        # )

        self.scratch.input_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0, groups=self.groups),
            nn.ReflectionPad2d(1),
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=0, groups=self.groups),
            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=0, groups=self.groups),
            nn.ReflectionPad2d(1),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=0, groups=self.groups)
        )

        self.pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)

    def forward(self, x):

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        # print(layer_1.shape)
        # print(layer_2.shape)
        # print(layer_3.shape)
        # print(layer_4.shape)

        # out = self.scratch.input_conv(x)
        # print(out.shape)
        #
        # layer_1_rn = self.scratch.layer1_rn(out)
        # print(layer_1_rn.shape)
        # layer_2_rn = self.scratch.layer2_rn(layer_1_rn)
        # print(layer_2_rn.shape)
        # layer_3_rn = self.scratch.layer3_rn(layer_2_rn)
        # print(layer_3_rn.shape)
        # layer_4_rn = self.scratch.layer4_rn(layer_3_rn)
        # print(layer_4_rn.shape)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # print(layer_4_rn.shape)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print(path_1.shape)


        out = self.scratch.output_conv(path_1)
        out = out.clamp(0.2, 20)

        # print(out.shape)

        return out



def MutiScaleLuminanceEstimation(img):
    print(img.shape)
    sigma_list = [15, 60, 90]
    img = img.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    print(img.shape)
    w, h, c = img.shape
    img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3)
    Luminance = np.ones_like(img).astype(np.float)
    for sigma in sigma_list:
        Luminance1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        Luminance1 = np.clip(Luminance1, 0, 255)
        Luminance += Luminance1
    Luminance = Luminance / 3
    L = (Luminance - np.min(Luminance)) / (np.max(Luminance) - np.min(Luminance) + 0.0001)
    L = np.uint8(L * 255)
    L = cv2.resize(L, dsize=(h, w))
    return L


if __name__ == "__main__":

    img = Image.open('/home/whealther/whealther230224/2.jpg')
    image1 = cv2.imread('/home/whealther/whealther230224/1.jpg')
    # print(img.size)
    img = transforms.functional.to_tensor(img)
    img = img.unsqueeze(0)
    image1 = transforms.functional.to_tensor(image1)
    image1 = image1.unsqueeze(0)
    # print(img.shape)
    # print(img.shape)
    # print(image1.shape)

    # create depthNet
    net = DepthEstimationNet()
    image = net(img)
    print(image.shape)
    # print("the image that was created by the DepthNet is:")
    # print(image.shape)

    # create the light field map
    imagel = MutiScaleLuminanceEstimation(image1)
    imagel = Image.fromarray(imagel)
    print(imagel.shape)

    # cat
    # x = torch.cat([imagel, image, img], 1)
    # print(x.shape)
    # x = torch.nn.Conv2d(7, 3, kernel_size=3, stride=1, padding=1, padding_mode='valid')(x)
    # print(x.shape)

    # save image
    image = image.squeeze(0)
    toPIL = transforms.ToPILImage()
    image = toPIL(image)
    print('-----save------')
    image.save('/media/whealther/diskb/UW-CycleGAN-main/dataset/depth_light/2depth.png')
    imagel.save('/media/whealther/diskb/UW-CycleGAN-main/dataset/depth_light/2light.png')
    print('have been saved')


