import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from torchvision.models.vgg import vgg19, vgg16
from torchvision.models.resnet import resnet50
import torch.nn as nn



# 感知一致性损失，计算X和G(X)的多层次特征之间的差异，以训练网络保留其图像细节
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        vgg = vgg19(pretrained=True).eval()

        # vgg的1,3,5,9,13层的输出
        # self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval()
        # self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval()
        self.loss_net10 = nn.Sequential(*list(vgg.features)[:10]).eval()  # maxpool2
        self.loss_net36 = nn.Sequential(*list(vgg.features)[:36]).eval()  # maxpool5
        # self.loss_net13 = nn.Sequential(*list(vgg.features)[:13]).eval()

    def forward(self, x, y):
        # loss1 = self.mse(self.loss_net1(x), self.loss_net1(y))
        # loss3 = self.mse(self.loss_net3(x), self.loss_net3(y))
        # loss5 = self.mse(self.loss_net5(x), self.loss_net5(y))
        # loss9 = self.mse(self.loss_net9(x), self.loss_net9(y))
        # loss13 = self.mse(self.loss_net13(x), self.loss_net13(y))
        loss10 = self.mse(self.loss_net10(x), self.loss_net10(y))
        loss36 = self.mse(self.loss_net36(x), self.loss_net36(y))

        loss = loss10 * 0.5 + loss36 * 0.5

        # loss = loss1 * 0.2 + loss3 * 0.2 + loss5 * 0.2 + loss9 * 0.2 + loss13 * 0.2
        return loss
