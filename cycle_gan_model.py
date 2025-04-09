import torch
import itertools
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .loss_380 import *
from .perceptual import PerceptualLoss
# from .loss_focal_frequency import FocalFrequencyLoss as FFL
from .create_light import MutiScaleLuminanceEstimationFu
from .create_depth import DepthEstimationNet
from .frequency import HighPassFilterMethod, LowPassFilterMethod


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'perceptual_A', 'perceptual_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_B = networks.define_Gen(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        pretrained_model = torch.load('./weight/latest_net_G_A.pth')
        self.netG_B.load_state_dict(pretrained_model.state_dict(), strict=False)


        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.perceptual = PerceptualLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.real_A, self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_A, self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        self.netG_B.requires_grad_(True)
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_perceptual = 4
        lambda_380 = 0.1
        lambda_FFL = 0.001
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A, self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # perceptual
        self.loss_perceptual_A = self.perceptual(self.real_A, self.fake_B) * lambda_perceptual
        self.loss_perceptual_B = self.perceptual(self.real_B, self.fake_A) * lambda_perceptual
        #loss_380
        # self.loss_380A = loss380_U(self.real_A, self.rec_A, self.fake_B) * lambda_380
        # self.loss_380B = loss380_A(self.real_B, self.rec_B, self.fake_A) * lambda_380
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_perceptual_B + self.loss_perceptual_A
        self.set_requires_grad(self.netG_B, False)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights









# class CycleGANModel(BaseModel):
#     """
#     This class implements the CycleGAN model, for learning image-to-image translation without paired data.
#
#     The model training requires '--dataset_mode unaligned' dataset.
#     By default, it uses a '--netG resnet_9blocks' ResNet generator,
#     a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
#     and a least-square GANs objective ('--gan_mode lsgan').
#
#     CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
#     """
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         """Add new dataset-specific options, and rewrite default values for existing options.
#
#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
#
#         Returns:
#             the modified parser.
#
#         For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
#         A (source domain), B (target domain).
#         Generators: G_A: A -> B; G_B: B -> A.
#         Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
#         Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
#         Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
#         Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
#         Dropout is not used in the original CycleGAN paper.
#         """
#         parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
#         if is_train:
#             parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
#             parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
#             parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
#
#         return parser
#
#
#     def __init__(self, opt):
#         """Initialize the CycleGAN class.
#
#         Parameters:
#             opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseModel.__init__(self, opt)
#         # 定义网络的损失函数，本网络的损失函数有：光场损失LightLoss、高频损失FH与低频损失FL、水下暗通道损失UDCLoss
#         self.loss_names = ['light', 'depth', 'perceptual', 'frequency', 'consistent']
#         # 保存的图像信息
#         # self.retention_image, self.depth, self.light, self.retention_light, self.retention_depth, self.retention_high, self.retention_low, self.clean_high, self.clean_low
#         self.visual_names = ['real_A', 'real_B', 'retention_image', 'depth', 'light', 'retention_light', 'retention_depth', 'retention_high', 'retention_low', 'underwater_high', 'underwater_low']
#         # self.visual_names = ['clean_image', 'light_image', 'depth_image', 'retention_image', 'retention_depth', 'retention_light', 'retention_high', 'retention_low', 'clean_high', 'clean_low']
#         if self.isTrain:
#             self.model_names = ['G_A']
#         else:
#             self.model_names = ['G_A']
#
#         # 定义生成网络
#         # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
#                                         # not opt.no_dropout, opt.init_type, opt.init_gain, opt.device)
#         self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         if self.isTrain:
#             self.retention_pool = ImagePool(opt.pool_size)
#
#         # 定义损失函数的基础运算函数
#         self.LightLoss = torch.nn.L1Loss()
#         self.DepthLoss = torch.nn.L1Loss()
#         self.PerceptualLoss = PerceptualLoss().to(self.device)
#         self.ImageFrequencyLoss = torch.nn.L1Loss()
#         self.ConLoss = torch.nn.L1Loss()
#
#         # 定义优化器
#         self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#         self.optimizers.append(self.optimizer_G)
#
#     def set_input(self, input):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.
#
#         Parameters:
#             input (dict): include the data itself and its metadata information.
#
#         The option 'direction' can be used to swap domain A and domain B.
#         """
#         # A是水下图像， B是空中图像; AtoB为水下图像增强
#         AtoB = self.opt.direction == 'AtoB'
#         self.real_A = input['A' if AtoB else 'B'].to(self.device)
#         self.real_B = input['B' if AtoB else 'A'].to(self.device)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']
#
#     def forward(self):
#
#         # 水下图像增强
#         self.retention_image, self.depth, self.light, self.retention_light, self.retention_depth, self.retention_high, self.retention_low, self.underwater_high, self.underwater_low = self.netG_A(self.real_A, self.real_B)  # clean underwater
#         # self.retention_image, self.depth, self.light, self.retention_light, self.retention_depth, self.retention_hfre, self.retention_lfre, self.clean_hfre, self.clean_lfre = self.netG_A(self.real_A, self.real_B)  # clean underwater
#
#     def backward_G(self):
#
#         lambda_depth = 1
#         lambda_light = 1
#         lambda_frequency = 0.01
#         lambda_perceptual = 40
#         lambda_consistent = 1000
#
#
#         self.loss_consistent = self.ConLoss(self.retention_image, self.real_B)
#         self.loss_light = self.LightLoss(self.retention_light, self.light)
#         self.loss_depth = self.DepthLoss(self.retention_depth, self.depth)
#         self.loss_perceptual = self.PerceptualLoss(self.retention_image, self.real_A)
#         self.loss_frequency = self.ImageFrequencyLoss(self.retention_high, self.underwater_high) + self.ImageFrequencyLoss(self.retention_low, self.underwater_low)
#
#
#         self.loss_G = self.loss_light * lambda_light + self.loss_depth * lambda_depth + self.loss_perceptual * lambda_perceptual + self.loss_frequency * lambda_frequency + self.loss_consistent * lambda_consistent
#         self.loss_G.backward()
#
#
#     def optimize_parameters(self):
#
#         # forward
#         self.forward()      # compute fake images and reconstruction images.
#         self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
#         self.backward_G()             # calculate gradients for G_A and G_B
#         self.optimizer_G.step()       # update G_A and G_B's weights


