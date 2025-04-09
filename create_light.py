import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import reduce
from torch.autograd import Variable
# from utils.blocks import FeatureFusionBlock_custom, Interpolate, _make_encoder, _make_scratch, _make_pretrained_efficientnet_lite3
# from .blocks import FeatureFusionBlock_custom, Interpolate, _make_encoder, _make_scratch, _make_pretrained_efficientnet_lite3
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2

class MutiScaleLuminanceEstimation(nn.Module):

    def forward(self, img, device):
        sigma_list = [15, 60, 90]
        # print(img.shape)
        # img = img.numpy()
        img = img.mul(255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        # img = img.cuda().numpy().squeeze(0).transpose((1, 2, 0))
        w, h, c = img.shape
        # print(img.shape)
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
        L = transforms.functional.to_tensor(L)
        L = L.unsqueeze(0)
        L = L.to(device)
        return L


class MutiScaleLuminanceEstimationFu(nn.Module):

    def forward(self, img):
        sigma_list = [15, 60, 90]
        # print(img.shape)
        # img = img.numpy()
        img = img.mul(255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        # img = img.cuda().numpy().squeeze(0).transpose((1, 2, 0))
        w, h, c = img.shape
        # print(img.shape)
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
        L = transforms.functional.to_tensor(L)
        L = L.unsqueeze(0)

        return L



def MutiScaleLuminanceEstimationFu(img, device):
    sigma_list = [15, 60, 90]
    # print(img.shape)
    img = img.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    w, h, c = img.shape
    # print(img.shape)
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
    L = transforms.functional.to_tensor(L)
    L = L.unsqueeze(0)
    L = L.to(int(device))
    return L