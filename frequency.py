import numpy as np
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Conv2d

class LowPassFilter(nn.Module):
    def forward(self, image):
        image = image.mul(255).byte()
        image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        fre = np.fft.fft2(image)
        fre_shift = np.fft.fftshift(fre)

        rows, cols = image.shape
        crows, ccols = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols))
        mask[crows - 10:crows + 10, ccols - 10:ccols + 10] = 1
        f = fre_shift * mask
        img_low = np.abs(np.fft.ifft2(f))

        return img_low

class HighPassFilter(nn.Module):
    def forward(self, image):
        image = image.mul(255).byte()
        image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        fre = np.fft.fft2(image)
        fre_shift = np.fft.fftshift(fre)

        rows, cols = image.shape
        crows, ccols = int(rows / 2), int(cols / 2)

        # 高频图像
        mask = np.ones((rows, cols))
        mask[crows - 10:crows + 10, ccols - 10:ccols + 10] = 0
        f = fre_shift * mask

        img_high = np.abs(np.fft.ifft2(f))

        return img_high



def LowPassFilterMethod(image):

    image = image.mul(255).byte()
    image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    fre = np.fft.fft2(image)
    fre_shift = np.fft.fftshift(fre)

    rows, cols, n = image.shape
    crows, ccols = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols))
    mask[crows - 10:crows + 10, ccols - 10:ccols + 10] = 1
    f = fre_shift * mask
    img_low = np.abs(np.fft.ifft2(f))

    return img_low


def HighPassFilterMethod(image):

    image = image.mul(255).byte()
    image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    fre = np.fft.fft2(image)
    fre_shift = np.fft.fftshift(fre)

    rows, cols = image.shape
    crows, ccols = int(rows / 2), int(cols / 2)

    # 高频图像
    mask = np.ones((rows, cols))
    mask[crows - 10:crows + 10, ccols - 10:ccols + 10] = 0
    f = fre_shift * mask

    img_high = np.abs(np.fft.ifft2(f))

    return img_high

# class FrequencyLoss(nn.Module):
#     def __init__(self):
#         super
#         self.highfilter = HighPassFilter
#         self.lowfilter = LowPassFilter
#         self.loss = nn.L1Loss
#         self.alpha = 1
#         self.beta = 1
#
#     def forward(self, x, y):
#
#         x_high = self.highfilter(x)
#         y_high = self.highfilter(y)
#
#         x_low = self.lowfilter(x)
#         y_low = self.lowfilter(y)
#
#         high_loss = self.loss(x_high, y_high)
#         low_loss = self.loss(x_low, y_low)
#
#         loss = high_loss * self.alpha + low_loss * self.beta
#
#         return loss

#
#
# image = cv2.imread('/media/whealther/张健老师的教学材料/test0728/1.png', 0)
# print(image.shape)
# fre = np.fft.fft2(image)
# fre_shift = np.fft.fftshift(fre)
#
# rows, cols = image.shape
# crows, ccols = int(rows/2), int(cols/2)
#
# # 高频图像
# mask = np.ones((rows, cols))
# mask[crows-10:crows+10, ccols-10:ccols+10] = 0
# f = fre_shift * mask
# img_high = np.abs(np.fft.ifft2(f))
# print(img_high.shape)
# cv2.imwrite('/home/whealther/whealther230224/high1.jpg', img_high)
#
#
# # 低频图像
# mask = np.zeros((rows, cols))
# mask[crows-10:crows+10, ccols-10:ccols+10] = 1
# f = fre_shift * mask
# img_low = np.abs(np.fft.ifft2(f))
# cv2.imwrite('/home/whealther/whealther230224/low1.jpg', img_low)
#
#
# # 高频图像
# fre_shift[crows-10:crows+10, ccols-10:ccols+10] = 0
#
# ishift = np.fft.ifftshift(fre_shift)
# img_high = np.fft.ifft2(ishift)
# img_high = np.abs(img_high)
#
# cv2.imwrite('/home/whealther/whealther230224/high2.jpg', img_high)
#
# # 低频图像
# dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
# dftshift = np.fft.fftshift(dft)
#
# cows, cols = image.shape
# crows, ccol = int(cows / 2), int(cols / 2)
# mask = np.zeros((cows, cols, 2), np.uint8)
# mask[crows-20:crows+20, ccol-20:ccol+20] = 1
#
# idftshift = dftshift * mask
#
#
# image = np.abs(np.fft.ifft2(idftshift))
# # f = np.fft.ifftshift(idftshift)
# # idft = cv2.idft(f)
# # img_low = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
#
# cv2.imwrite('/home/whealther/whealther230224/low2.jpg', img_low)
#
#
# #
# #
# # # 低频图像
# # dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
# # dftshift = np.fft.fftshift(dft)
# #
# # cows, cols = image.shape
# # mask = np.zeros((cows, cols, 2), np.uint8)
# # mask[int(cows/2-10):int(cows/2+10), int(cols/2-10):int(cols/2+10)] = 1
# #
# # idftshift = dftshift * mask
# #
# # idftshift = np.fft.ifftshift(idftshift)
# # idft = cv2.idft(idftshift)
# # img_low = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
# #
# # cv2.imwrite('/home/whealther/whealther230224/low2.jpg', img_low)
# #
