import torch
import cv2
import argparse
import torch.nn.functional as F

def get_gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

def get_low_freq(im, gauss_kernel):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = get_gaussian_blur(im, gauss_kernel, padding=padding)
    return low_freq


def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel


def find_fake_freq(im, gauss_kernel, index=None):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = gaussian_blur(im, gauss_kernel, padding=padding)
    im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    return torch.cat((low_freq, im_gray - low_gray),1)


def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    # fft = torch.rfft(image, 2, onesided=False)
    # fft = torch.fft.fft(image,2)
    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft = torch.stack((fft.real, fft.imag), -1)
    fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    return fft_mag


def fft_L1_loss(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def fft_L1_loss_mask(fake_image, real_image, mask):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft * mask, real_fft * mask)
    return loss


def fft_L1_loss_color(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def decide_circle(N=4, L=256, r=96, size=256):
    x = torch.ones((N, L, L))
    for i in range(L):
        for j in range(L):
            if (i - L / 2 + 0.5) ** 2 + (j - L / 2 + 0.5) ** 2 < r ** 2:
                x[:, i, j] = 0
    return x, torch.ones((N, L, L)) - x

'''
args:
    gauss_size
    radius
    batch
    size                (img_size)

    w_low_recon
    w_high_recon        used in 2 place

    w_recon_fft
    w_fft_swap_H
    
    input:
            tensor img
    output:
            loss_380
    
    
    real_img1 ==> fake_img1
    
    real_img2 ==> fake_img2 (style_trans_target)
    
    for cyclegan
        real_img1 = real_x
        fake_img1 = rec_x
        fake_img2 = fake_y
'''
def loss380_U(real_img1,fake_img1,fake_img2):

    # parser = argparse.ArgumentParser()
    #
    #
    # parser.add_argument("--size", type=int, default=256)
    # parser.add_argument('--gauss_size', type=int, default=21)
    # parser.add_argument('--w_high_recon', type=float, default=1)
    # parser.add_argument('--w_low_recon', type=float, default=1)
    # parser.add_argument('--radius', type=int, default=21)
    # parser.add_argument('--w_recon_fft', type=float, default=1)
    # parser.add_argument('--w_fft_swap_H', type=float, default=1)
    #
    # args = parser.parse_args()
    size = 256
    gauss_size_U = 39
    gauss_size_A = 21
    w_high_recon=1
    w_low_recon = 1
    radius=21
    w_recon_fft =1
    w_fft_swap_H =1


    gauss_kernel_U = get_gaussian_kernel(gauss_size_U).cuda()
    gauss_kernel_A = get_gaussian_kernel(gauss_size_A).cuda()
    mask_h, mask_l =decide_circle(r=radius, N=1, L=size)
    mask_h, mask_l = mask_h.cuda(), mask_l.cuda()


    real_img_freq1 = find_fake_freq(real_img1, gauss_kernel_U)
    fake_img1_freq = find_fake_freq(fake_img1, gauss_kernel_U)
    fake_img2_freq = find_fake_freq(fake_img2, gauss_kernel_A)

    recon_freq_loss_img1_low = F.l1_loss(fake_img1_freq[:, :3, :, :], real_img_freq1[:, :3, :, :])
    recon_freq_loss_img1_high = F.l1_loss(fake_img1_freq[:, 3:6, :, :], real_img_freq1[:, 3:6, :, :])


    recon_fft = fft_L1_loss_color(fake_img1, real_img1)                                                                 #eq.8(L_rec_fft)    w=args.w_recon_fft=1
    recon_freq_loss_img1 =w_low_recon * recon_freq_loss_img1_low + w_high_recon * recon_freq_loss_img1_high             #eq.4(L_rec_pix)    w=1
    recon_freq_loss_img2_structure = F.l1_loss(fake_img2_freq[:, 3:6, :, :], real_img_freq1[:, 3:6, :, :])              #eq.5(L_trans_pix)  w=args.w_high_recon=1
    fft_swap_H =  fft_L1_loss_mask(fake_img2, real_img1, mask_h)                                                        #eq.9(L_trans_fft)  w=args.w_fft_swap_H=1

    loss_380 = recon_freq_loss_img1 + w_high_recon * recon_freq_loss_img2_structure
    + w_recon_fft * recon_fft + w_fft_swap_H * fft_swap_H

    return loss_380

def loss380_A(real_img1,fake_img1,fake_img2):

    # parser = argparse.ArgumentParser()
    #
    #
    # parser.add_argument("--size", type=int, default=256)
    # parser.add_argument('--gauss_size', type=int, default=21)
    # parser.add_argument('--w_high_recon', type=float, default=1)
    # parser.add_argument('--w_low_recon', type=float, default=1)
    # parser.add_argument('--radius', type=int, default=21)
    # parser.add_argument('--w_recon_fft', type=float, default=1)
    # parser.add_argument('--w_fft_swap_H', type=float, default=1)
    #
    # args = parser.parse_args()
    size = 256
    gauss_size_U = 39
    gauss_size_A = 21
    w_high_recon=1
    w_low_recon = 1
    radius=21
    w_recon_fft =1
    w_fft_swap_H =1


    gauss_kernel_U = get_gaussian_kernel(gauss_size_U).cuda()
    gauss_kernel_A = get_gaussian_kernel(gauss_size_A).cuda()
    mask_h, mask_l =decide_circle(r=radius, N=1, L=size)
    mask_h, mask_l = mask_h.cuda(), mask_l.cuda()


    real_img_freq1 = find_fake_freq(real_img1, gauss_kernel_A)
    fake_img1_freq = find_fake_freq(fake_img1, gauss_kernel_A)
    fake_img2_freq = find_fake_freq(fake_img2, gauss_kernel_U)

    recon_freq_loss_img1_low = F.l1_loss(fake_img1_freq[:, :3, :, :], real_img_freq1[:, :3, :, :])
    recon_freq_loss_img1_high = F.l1_loss(fake_img1_freq[:, 3:6, :, :], real_img_freq1[:, 3:6, :, :])


    recon_fft = fft_L1_loss_color(fake_img1, real_img1)                                                                 #eq.8(L_rec_fft)    w=args.w_recon_fft=1
    recon_freq_loss_img1 =w_low_recon * recon_freq_loss_img1_low + w_high_recon * recon_freq_loss_img1_high             #eq.4(L_rec_pix)    w=1
    recon_freq_loss_img2_structure = F.l1_loss(fake_img2_freq[:, 3:6, :, :], real_img_freq1[:, 3:6, :, :])              #eq.5(L_trans_pix)  w=args.w_high_recon=1
    fft_swap_H =  fft_L1_loss_mask(fake_img2, real_img1, mask_h)                                                        #eq.9(L_trans_fft)  w=args.w_fft_swap_H=1

    loss_380 = recon_freq_loss_img1 + w_high_recon * recon_freq_loss_img2_structure
    + w_recon_fft * recon_fft + w_fft_swap_H * fft_swap_H

    return loss_380
