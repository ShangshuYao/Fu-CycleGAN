import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import scipy.stats as st
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.autograd as autograd
from math import exp
import torchvision.transforms as t
#import pytorch_ssim



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
#     # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
#     if val_range is None:
#         if torch.max(img1) > 128:
#             max_val = 255
#         else:
#             max_val = 1
#
#         if torch.min(img1) < -0.5:
#             min_val = -1
#         else:
#             min_val = 0
#         L = max_val - min_val
#     else:
#         L = val_range
#     padd = 0
#     (_, channel, height, width) = img1.size()
#     if window is None:
#         real_size = min(window_size, height, width)
#         window = create_window(real_size, channel=channel).to(img1.device)
#
#     mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
#
#     C1 = (0.01 * L) ** 2
#     C2 = (0.03 * L) ** 2
#
#     v1 = 2.0 * sigma12 + C2
#     v2 = sigma1_sq + sigma2_sq + C2
#     cs = torch.mean(v1 / v2)  # contrast sensitivity
#
#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
#
#     if size_average:
#         ret = ssim_map.mean()
#     else:
#         ret = ssim_map.mean(1).mean(1).mean(1)
#
#     if full:
#         return ret, cs
#     return ret
# #
# # def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
# #     device = img1.device
# #     weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
# #     # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
# #     levels = weights.size()[0]
# #     mssim = []
# #     mcs = []
# #     for _ in range(levels):
# #         sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
# #         #print("sim",sim)
# #         mssim.append(sim)
# #         mcs.append(cs)
# #
# #         img1 = F.avg_pool2d(img1, (2, 2))
# #         img2 = F.avg_pool2d(img2, (2, 2))
# #
# #     mssim = torch.stack(mssim)
# #     mcs = torch.stack(mcs)
# #
# #     # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
# #     if normalize:
# #         mssim = (mssim + 1) / 2
# #         mcs = (mcs + 1) / 2
# #
# #     pow1 = mcs ** weights
# #     pow2 = mssim ** weights
# #     # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
# #     output = torch.prod(pow1[:-1] * pow2[-1])
# #     return output
# #
# # # Classes to re-use window
# class SSIM(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True, val_range=None):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.val_range = val_range
#
#         # Assume 1 channel for SSIM
#         self.channel = 1
#         self.window = create_window(window_size)
#
#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#
#         if channel == self.channel and self.window.dtype == img1.dtype:
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
#             self.window = window
#             self.channel = channel
#
#         return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

# class MSSSIM(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True, channel=3):
#         super(MSSSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = channel
#
#     def forward(self, img1, img2):
#         # TODO: store window between calls if possible,
#         # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
#         return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)

def white_balance(img):
    '''
    灰度世界假设
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    img_blur = t.GaussianBlur(21,sigma=(0.1,2.0))  # 高斯模糊化
    img_blur = img_blur(img)

    B, G, R = np.double(img_blur.data.cpu()[:, :, 0]), np.double(img_blur.data.cpu()[:, :, 1]), np.double(img_blur.data.cpu()[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)   #RGB三通道的均值
    K = (B_ave + G_ave + R_ave) / 3   #RGB均值的通道平均值
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave     #通道均值占各自通道均值的比重
    a = np.array([Kb, Kg, Kr])  #向量化
    return a

class Cal_Oushi_loss(nn.Module):
    def __init__(self,cuda=True):   #
        super(Cal_Oushi_loss, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, img1, img2):
        img1_blur = white_balance(img1)  # 白平衡
        img2_blur = white_balance(img2)
        img1_tensor = torch.tensor(img1_blur)
        img2_tensor = torch.tensor(img2_blur)
        # pdist = nn.PairwiseDistance(p=2)  # 欧几里得距离
        # s= pdist(img1_tensor,img2_tensor)
        #mse = torch.nn.MSELoss()           #均方差
        #loss = mse(img1_tensor,img2_tensor)
        mae = torch.nn.L1Loss()             #l1
        loss = mae(img1_tensor, img2_tensor)
        return loss

# class MS_SSIM_L1_LOSS(nn.Module):
#     """
#     Have to use cuda, otherwise the speed is too slow.
#     Both the group and shape of input image should be attention on.
#     I set 255 and 1 for gray image as default.
#     """
#     def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
#                  data_range=255.0,
#                  K=(0.01, 0.03),  # c1,c2
#                  alpha=0.025,  # weight of ssim and l1 loss
#                  compensation=200.0,  # final factor for total loss
#                  cuda_dev=0,  # cuda device choice
#                  channel=3):  # RGB image should set to 3 and Gray image should be set to 1
#         super(MS_SSIM_L1_LOSS, self).__init__()
#         self.channel = channel
#         self.DR = data_range
#         self.C1 = (K[0] * data_range) ** 2
#         self.C2 = (K[1] * data_range) ** 2
#         self.pad = int(2 * gaussian_sigmas[-1])
#         self.alpha = alpha
#         self.compensation = compensation
#         filter_size = int(4 * gaussian_sigmas[-1] + 1)
#         g_masks = torch.zeros(
#             (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
#         for idx, sigma in enumerate(gaussian_sigmas):
#             if self.channel == 1:
#                 # only gray layer
#                 g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#             elif self.channel == 3:
#                 # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
#                 g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
#                                                                                    sigma)  # 每层mask对应不同的sigma
#                 g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#                 g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#             else:
#                 raise ValueError
#         self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型
#
#     def _fspecial_gauss_1d(self, size, sigma):
#         """Create 1-D gauss kernel
#         Args:
#             size (int): the size of gauss kernel
#             sigma (float): sigma of normal distribution
#
#         Returns:
#             torch.Tensor: 1D kernel (size)
#         """
#         coords = torch.arange(size).to(dtype=torch.float)
#         coords -= size // 2
#         g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#         g /= g.sum()
#         return g.reshape(-1)
#
#     def _fspecial_gauss_2d(self, size, sigma):
#         """Create 2-D gauss kernel
#         Args:
#             size (int): the size of gauss kernel
#             sigma (float): sigma of normal distribution
#
#         Returns:
#             torch.Tensor: 2D kernel (size x size)
#         """
#         gaussian_vec = self._fspecial_gauss_1d(size, sigma)
#         return torch.outer(gaussian_vec, gaussian_vec)
#         # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
#         # then out must be a matrix of size (n \times m)(n×m).
#
#     def forward(self, x, y):
#         b, c, h, w = x.shape
#         assert c == self.channel
#
#         mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
#         muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度
#
#         mux2 = mux * mux
#         muy2 = muy * muy
#         muxy = mux * muy
#
#         sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
#         sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
#         sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy
#
#         # l(j), cs(j) in MS-SSIM
#         l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
#         cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
#         if self.channel == 3:
#             lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
#             PIcs = cs.prod(dim=1)
#         elif self.channel == 1:
#             lM = l[:, -1, :, :]
#             PIcs = cs.prod(dim=1)
#
#         loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]
#
#         loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
#         # average l1 loss in num channels
#         gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
#                                groups=c, padding=self.pad).mean(1)  # [B, H, W]
#
#         loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
#         loss_mix = self.compensation * loss_mix
#
#         return loss_mix.mean()
#
#
# class VGG19_PercepLoss(nn.Module):
#     """ Calculates perceptual loss in vgg19 space
#     """
#     def __init__(self, _pretrained_=True):
#         super(VGG19_PercepLoss, self).__init__()
#         self.vgg = models.vgg19(pretrained=_pretrained_).features
#         for param in self.vgg.parameters():
#             param.requires_grad_(False)
#
#     def get_features(self, image, layers=None):
#         if layers is None:
#             layers = {'31': 'relu5_2'} # may add other layers
#         features = {}
#         x = image
#         for name, layer in self.vgg._modules.items():
#             x = layer(x)
#             if name in layers:
#                 features[layers[name]] = x
#         return features
#
#     def forward(self, pred, true, layer='relu5_2'):
#         true_f = self.get_features(true)
#         pred_f = self.get_features(pred)
#         return torch.mean((true_f[layer]-pred_f[layer])**2)
#
# class Gradient_Penalty(nn.Module):
#     """ Calculates the gradient penalty loss for WGAN GP
#     """
#     def __init__(self, cuda=True):
#         super(Gradient_Penalty, self).__init__()
#         self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
#     def forward(self, D, real, fake):
#         # Random weight term for interpolation between real and fake samples
#         eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
#         # Get random interpolation between real and fake samples
#         interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
#         d_interpolates = D(interpolates)
#         fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
#         # Get gradient w.r.t. interpolates
#         gradients = autograd.grad(outputs=d_interpolates,
#                                   inputs=interpolates,
#                                   grad_outputs=fake,
#                                   create_graph=True,
#                                   retain_graph=True,
#                                   only_inputs=True,)[0]
#         gradients = gradients.view(gradients.size(0), -1)
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#         return gradient_penalty
#
# def gauss_kernel(kernlen=21, nsig=3, channels=1):
#     interval = (2 * nsig + 1.) / (kernlen)
#     x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#     kernel = kernel_raw / kernel_raw.sum()
#     out_filter = np.array(kernel, dtype=np.float32)
#     out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
#     out_filter = np.repeat(out_filter, channels, axis=2)
#     return out_filter
#
# class ColorLoss(nn.Module):
#     def __init__(self, nc=3):
#         super(ColorLoss, self).__init__()
#         self.nc = nc
#         kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
#         kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#         # self.cc = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
#
#     def forward(self, x1, x2):
#         if x1.size(1) != self.nc:
#             raise RuntimeError(
#                 "The channel of input [%d] does not match the preset channel [%d]" % (x1.size(1), self.nc))
#         x1 = F.conv2d(x1, self.weight, stride=1, padding=10, groups=self.nc)
#
#         if x2.size(1) != self.nc:
#             raise RuntimeError(
#                 "The channel of input [%d] does not match the preset channel [%d]" % (x2.size(1), self.nc))
#         x2 = F.conv2d(x2, self.weight, stride=1, padding=10, groups=self.nc)
#         s = F.mse_loss(x1, x2)
#         return s

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
            parser.add_argument('--lambda_A', type=float, default=10, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_C', type=float, default=0, help='weight for color loss (A -> B -> A)')
            parser.add_argument('--lambda_D', type=float, default=5, help='weight for content loss (B -> A -> B)')   #5
            parser.add_argument('--lambda_E', type=float, default=0, help='weight for msssim loss' )
            parser.add_argument('--lambda_F', type=float, default=0, help='weight for focal loss (B -> A -> B)')
            parser.add_argument('--lambda_col', type=float, default=0.5, help='weight for col loss (A -> B -> A)')  #0.5
            # parser.add_argument('--lambda_H', type=float, default=10.0, help='weight for MSSSIM loss (A -> B -> A)')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B','content_A',  'content_B']  #, 'content_A',  'content_B', 'color_A', 'color_B', 'ms_A', 'ms_B','fc_A','fc_B'
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
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

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
            #self.criterionIdt = torch.nn.L1Loss()
            self.criterionColor = Cal_Oushi_loss().to(self.device)   #对比度损失
            #self.criterionCon = VGG19_PercepLoss() .to(self.device)  #内容损失
            #self.criterionCon = SSIM().to(self.device) #内容损失
            self.criterionCon = torch.nn.L1Loss() #内容损失
            #self.criterionCon = MS_SSIM_L1_LOSS().to(self.device)  #内容损失
            #self.criterionGp = Gradient_Penalty().to(self.device)    #梯度惩罚损失
           #self.criterionMS = MSSSIM().to(self.device)   #结构相似度损失
            #self.criterionF = torch.nn.BCEWithLogitsLoss()   #像素级损失
            #self.criterionIOU = networks.IOU().to(self.device)   #特征图级损失
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
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
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
        lambda_col = self.opt.lambda_col
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_col = self.criterionColor(real, fake)
        #loss_gp = self.criterionGp(netD, real, fake)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5+ loss_col * lambda_col# + loss_col * lambda_col
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
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # lambda_C = self.opt.lambda_C
        lambda_D = self.opt.lambda_D
        #lambda_E = self.opt.lambda_E
        #lambda_F = self.opt.lambda_F
        # lambda_G = self.opt.lambda_G
        # lambda_H = self.opt.lambda_H
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
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
        #color loss
        # self.loss_color_A = self.criterionColor(self.fake_B ,self.real_B) * lambda_C    # real_B, self.fake_B
        # self.loss_color_B = self.criterionColor(self.fake_A, self.real_A) * lambda_C

        self.con_A = self.netG_A(self.real_B)
        self.loss_content_A = self.criterionCon(self.con_A,self.real_B) * lambda_D    #self.fake_B,self.real_B
        self.con_B = self.netG_B(self.real_A)
        self.loss_content_B = self.criterionCon(self.con_B,self.real_A) * lambda_D
        # content loss
        # self.loss_content_A = self.criterionCon(self.fake_B, self.real_A) * lambda_D    #self.real_A, self.fake_B
        # self.loss_content_B = self.criterionCon(self.fake_A, self.real_B) * lambda_D
        #MSSSIM loss
        #self.loss_ms_A = self.criterionMS(self.real_B, self.fake_B) * lambda_E   #self.fake_B, self.real_A
        #self.loss_ms_B = self.criterionMS(self.fake_A, self.real_A) * lambda_E
        #focal loss

        #self.loss_fc_A = self.criterionF(self.fake_B, self.real_B) * lambda_F
        #self.loss_fc_B = self.criterionF(self.fake_A, self.real_A) * lambda_F


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_content_A + self.loss_content_B#+self.loss_color_A + self.loss_color_B + self.loss_content_A + self.loss_content_B
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


