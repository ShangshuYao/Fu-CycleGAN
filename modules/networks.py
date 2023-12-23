import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F
import math
from .layers import unetConv2,UNetBlock,unetUp
from .init_weight import init_weights


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        #net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, bn_size=4, growth_rate=32)
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
       # net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, bn_size=4, growth_rate=32)
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Full':
        net = FullGenerator(input_nc, output_nc,ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'FullU':
        net = FullUGenerator(input_nc, output_nc, ngf, num_layers=4, norm_layer=norm_layer, growth_rate=32,  bn_size=4, compression_rate=0.5, drop_rate=0)
    elif netG == 'FullUd':
        net = FullUdGenerator(input_nc, output_nc, ngf, is_deconv=True, is_batchnorm=True,num_input_features=128, num_layers=2, norm_layer=nn.BatchNorm2d, growth_rate=32,  bn_size=4, compression_rate=0.5, drop_rate=0)
    elif netG == 'ugan1_3':
        net = GeneratorUgan3(in_channels=3, output_nc=3)
    elif netG == 'UGAN_3':
        net = UNet3Plus(input_nc=3,output_nc=3, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # elif netG == 'UGAN_3a':
    #     net = UNet_3alex(input_nc, output_nc, feature_scale=4, is_deconv=True, is_batchnorm=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        # self.add_module("norm3", nn.BatchNorm2d(growth_rate))
        # self.add_module("relu3", nn.ReLU(inplace=True))
        # self.add_module("conv3", nn.Conv2d(growth_rate, growth_rate,
        #                                    kernel_size=1, stride=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate)
        # 在通道维上将输入和输出连结
        return torch.cat([x, new_features], 1)
#
class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1), layer)
#
# class _Transition(nn.Sequential):
#     """Transition layer between two adjacent DenseBlock"""
#     def __init__(self, num_input_feature, num_output_features):
#         super(_Transition, self).__init__()
#         self.add_module("norm", nn.BatchNorm2d(num_input_feature))
#         self.add_module("relu", nn.ReLU(inplace=True))
#         self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
#                                           kernel_size=1, stride=1, bias=False))
#         self.add_module("pool", nn.AvgPool2d(2, stride=2))

# class DenseNetGenerator(nn.Module):
#     "DenseNet-BC model"
#     def __init__(self, growth_rate=32, block_config=(3, 6, 12, 8), num_init_features=64,
#                  bn_size=4, compression_rate=0.5, drop_rate=0):
#         """
#         :param growth_rate: 增长率，即K=32
#         :param block_config: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
#         :param num_init_features: 第一个卷积的通道数一般为2*K=64
#         :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
#         :param compression_rate: 压缩因子
#         :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
#         :param num_classes: 分类数
#         """
#         super(DenseNetGenerator, self).__init__()
#         # first Conv2d
#         self.features = nn.Sequential(OrderedDict([
#             ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
#             ("norm0", nn.BatchNorm2d(num_init_features)),
#             ("relu0", nn.ReLU(inplace=True)),
#             ("pool0", nn.MaxPool2d(3, stride=2, padding=1))
#         ]))
#
#         # DenseBlock
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
#             self.features.add_module("denseblock%d" % (i + 1), block)
#             num_features += num_layers*growth_rate
#             if i != len(block_config) - 1:
#                 transition = _Transition(num_features, int(num_features*compression_rate))
#                 self.features.add_module("transition%d" % (i + 1), transition)
#                 num_features = int(num_features * compression_rate)
    #转置卷积
        # n_blocks = 5
        # for i in range(n_blocks):
        #     mult = 2** i
        #     self.features.add_module("conv3", nn.ConvTranspose2d(int(num_features/mult), int(num_features/(2**mult)), kernel_size=3, stride=2,padding=1,output_padding=1))
        #     self.features.add_module("norm3", nn.BatchNorm2d(int(num_features / (2**mult))))
        #     self.features.add_module("relu3", nn.ReLU(inplace=True))
        # self.features.add_module("conv3",
        #                          nn.ConvTranspose2d(int(num_features), int(num_features / 2), kernel_size=3,
        #                                             stride=2, padding=1, output_padding=1))
        # self.features.add_module("norm3", nn.BatchNorm2d(int(num_features / 2)))
        # self.features.add_module("relu3", nn.LeakyReLU(inplace=True))
        # self.features.add_module("conv4",
        #                          nn.ConvTranspose2d(int(num_features / 2), int(num_features / 4), kernel_size=3,
        #                                             stride=2, padding=1, output_padding=1))
        # self.features.add_module("norm4", nn.BatchNorm2d(int(num_features / 4)))
        # self.features.add_module("relu4", nn.LeakyReLU(inplace=True))
        # self.features.add_module("conv5",nn.ConvTranspose2d(int(num_features/4), int(num_features/8), kernel_size=3, stride=2,padding=1,output_padding=1))
        # self.features.add_module("norm5", nn.BatchNorm2d(int(num_features / 8)))
        # self.features.add_module("relu5", nn.LeakyReLU(inplace=True))
        # self.features.add_module("conv6", nn.ConvTranspose2d(int(num_features / 8), int(num_features / 16), kernel_size=3,
        #                                             stride=2, padding=1, output_padding=1))
        # self.features.add_module("norm6", nn.BatchNorm2d(int(num_features / 16)))
        # self.features.add_module("relu6", nn.LeakyReLU(inplace=True))
        # self.features.add_module("conv7",
        #                          nn.ConvTranspose2d(int(num_features / 16), int(num_features / 32), kernel_size=3,
        #                                             stride=2, padding=1, output_padding=1))
        # self.features.add_module("norm7", nn.BatchNorm2d(int(num_features / 32)))
        # self.features.add_module("relu7", nn.LeakyReLU(inplace=True))
        # # self.features.add_module("conv8",
        # #                          nn.ConvTranspose2d(int(num_features / 32), int(num_features / 64), kernel_size=3,
        # #                                             stride=2, padding=1, output_padding=1))
        # # self.features.add_module("norm8", nn.BatchNorm2d(int(num_features / 64)))
        # # self.features.add_module("relu8", nn.ReLU(inplace=True))
        #
        # # final bn+ReLU
        # #self.features.add_module("norm5", nn.BatchNorm2d(int(num_features/4)))
        # self.features.add_module("conv9", nn.Conv2d(int(num_features/32), 3, kernel_size=3, padding=1, stride=1))
        # self.features.add_module("relu9", nn.LeakyReLU(inplace=True))


    # def forward(self, x):
    #     out = self.features(x)
    #     return out

# class ResnetGenerator1(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
#
#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """
#
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', bn_size=4, growth_rate=32):
#         """Construct a Resnet-based generator
#
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         model = [nn.ReflectionPad2d(3),   #填充输入张量
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         n_downsampling = 2  #2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]
#
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks
#
#             model += [ResnetBlock(ngf * mult+i*growth_rate, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, bn_size=bn_size, growth_rate=growth_rate)]
#
#         model += [nn.Conv2d(ngf * mult + 6 * growth_rate, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
#             norm_layer(ngf * mult),
#             nn.ReLU(True)]
#
#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1,
#                                          bias=use_bias),
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.LeakyReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.LeakyReLU()]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self,  input):
#         """Standard forward"""
#         return self.model(input)


# class ResnetBlock1(nn.Sequential):
#     """Define a Resnet block"""
#
#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, bn_size, growth_rate):
#         """Initialize the Resnet block
#
#         A resnet block is a conv block with skip connections
#         We construct a conv block with build_conv_block function,
#         and implement skip connections in <forward> function.
#         Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
#         """
#         super(ResnetBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, growth_rate, bn_size)
#
#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, growth_rate, bn_size):
#         """Construct a convolutional block.
#
#         Parameters:
#             dim (int)           -- the number of channels in the conv layer.
#             padding_type (str)  -- the name of padding layer: reflect | replicate | zero
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#             use_bias (bool)     -- if the conv layer uses bias or not
#
#         Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
#         """
#         conv_block = []
#         p = 0
#
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#
#         conv_block += [nn.Conv2d(dim, bn_size*growth_rate, kernel_size=3, padding=p, bias=use_bias), norm_layer(bn_size*growth_rate), nn.ReLU(True)]  #3
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]
#         '''
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#         conv_block += [nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim//4), nn.ReLU(True)]
#         '''
#         #p=0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#         conv_block += [nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, padding=p, bias=use_bias), norm_layer(growth_rate)]  #3
#         self.relu = nn.ReLU()
#         return nn.Sequential(*conv_block)
#         #self.drop_rate = drop_rate
#
#     def forward(self, x):
#         # """Forward function (with skip connections)"""
#         # out = x + self.conv_block(x)  # add skip connections
#         # out = self.relu(out)
#         # return out
#         new_features = super(ResnetBlock, self).forward(x)
#         #new_features = self.relu(new_features)
#         # 在通道维上将输入和输出连结
#         return torch.cat([x, new_features], 1)


# class Res2netGenerator(nn.Module):
#     def __init__(self, block=Bottle2neck, layers=[3], baseWidth = 26, scale = 4):
#         self.inplanes = 64
#         super(Res2netGenerator, self).__init__()
#         self.baseWidth = baseWidth
#         self.scale = scale
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         #self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.relu4 = nn.LeakyReLU(inplace=True)
#         self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm2d(64)
#         self.relu5 = nn.LeakyReLU(inplace=True)
#         self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
#         self.bn6 = nn.BatchNorm2d(32)
#         self.relu6 = nn.LeakyReLU(inplace=True)
#         self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
#         self.bn7= nn.BatchNorm2d(16)
#         self.relu7 = nn.LeakyReLU(inplace=True)
#         self.conv8 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1)
#         self.bn8 = nn.BatchNorm2d(8)
#         self.relu8 = nn.LeakyReLU(inplace=True)
#         self.conv9 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1)
#         self.bn9 = nn.BatchNorm2d(4)
#         self.relu9 = nn.LeakyReLU(inplace=True)
#         self.conv10 = nn.ConvTranspose2d(4, 3, kernel_size=2, stride=2, padding=1)
#         self.bn10= nn.BatchNorm2d(3)
#         self.relu10 = nn.LeakyReLU(inplace=True)
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample=downsample,
#                         stype='stage', baseWidth = self.baseWidth, scale=self.scale))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         # x = self.layer2(x)
#         # x = self.layer3(x)
#         # x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu4(x)
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.relu5(x)
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.relu6(x)
#         x = self.conv7(x)
#         x = self.bn7(x)
#         x = self.relu7(x)
#         x = self.conv8(x)
#         x = self.bn8(x)
#         x = self.relu8(x)
#         x = self.conv9(x)
#         x = self.bn9(x)
#         x = self.relu9(x)
#         x = self.conv10(x)
#         x = self.bn10(x)
#         x = self.relu10(x)
#
#         #x = self.fc(x)
#         return x


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class FullUGenerator(nn.Module):
    """Full-based generator that consists of Fullnet + unet.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d, growth_rate=32,  bn_size=4, compression_rate=0.5, drop_rate=0):
        """Construct a U-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(FullUGenerator, self).__init__()
        model = [nn.Conv2d(input_nc, ngf*4, kernel_size=5, stride=1,padding=2),
                 norm_layer(ngf*4),
                 nn.LeakyReLU(True)]

        model += [nn.Conv2d(ngf*4, ngf*2, kernel_size=5, stride=1,padding=2),
                  norm_layer(ngf*2),
                  nn.LeakyReLU(True)]
        #num_features = num_input_features
        #model += _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
        n_sampling = 2
        for  i in range(n_sampling):
            mult = 2 ** i
            model += [nn.Conv2d(int(ngf*2/mult), int(ngf/mult), kernel_size=3, stride=1,padding=1),
                      norm_layer(int(ngf/mult)),
                      nn.LeakyReLU(True)]

        model += [nn.Conv2d(ngf//2, output_nc, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

        self.down1 = UNetDown(output_nc, 64, bn=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        #self.down5 = UNetDown(512, 512)   #  add
        self.down5 = UNetDown(512, 512, bn=False)
        # decoding layers
        self.up1 = UNetUp(512, 512)  # 256 256
        #self.up2 = UNetUp(1024, 512)  #
        self.up2 = UNetUp(1024, 256)  # 512 256
        self.up3 = UNetUp(512, 128)  # 512 128
        self.up4 = UNetUp(256, 64)  # 256 32
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1),
            nn.Tanh(),
        )
    def forward(self,  input):
        """Standard forward"""
        x = self.model(input)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        #d6 = self.down6(d5)  #  add
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        #u5 = self.up5(u4, d1)    #128
        return self.final(u4)

class FullUdGenerator(nn.Module):
    """Full-based generator that consists of FUllNet(DenseNet) + unet

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, is_deconv=True, is_batchnorm=True,num_input_features=128, num_layers=2, norm_layer=nn.BatchNorm2d, growth_rate=32,  bn_size=4, compression_rate=0.5, drop_rate=0):
        """Construct a -based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(FullUdGenerator, self).__init__()
        self.is_deconv = is_deconv
        #self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 512, 512, 512]

        model = [nn.Conv2d(output_nc, ngf, kernel_size=5, stride=1, padding=2),
                 norm_layer(ngf),
                 nn.LeakyReLU(True)]
        model += [nn.Conv2d(ngf, ngf * 2, kernel_size=5, stride=1, padding=2),
                  norm_layer(ngf * 2),
                  nn.LeakyReLU(True)]
        num_features = num_input_features
        model += _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
        n_sampling = 2
        for i in range(n_sampling):
            mult = 2 ** i
            model += [nn.Conv2d(int(ngf * 4 / mult), int(ngf * 2 / mult), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * 2 / mult)),
                      nn.LeakyReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=1, stride=1, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

        # downsampling
        self.conv1 = unetConv2(input_nc, filters[0], is_batchnorm=False)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.conv6 = unetConv2(filters[4], filters[5], self.is_batchnorm)
        self.cen = nn.Sequential(nn.Conv2d(filters[5], filters[6], 4, 2, 1),
                                 nn.ReLU(inplace=True))
        # upsampling
        self.up6 = unetUp(filters[6], filters[5], self.is_deconv)
        self.up5 = unetUp(1024, filters[4], self.is_deconv)
        self.up4 = unetUp(1024, filters[3], self.is_deconv)
        self.up3 = unetUp(1024, filters[2], self.is_deconv)
        self.up2 = unetUp(512, filters[1], self.is_deconv)
        self.up1 = unetUp(256, filters[0], self.is_deconv)
        self.out1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                  nn.Conv2d(filters[1], output_nc, 3, padding=1, bias=False),
                                  nn.Tanh())


        # self.down1 = UNetDown(output_nc, 64, bn=False)
        # self.down2 = UNetDown(64, 128)
        # self.down3 = UNetDown(128, 256)
        # self.down4 = UNetDown(256, 512)
        # self.down5 = UNetDown(512, 512, bn=False)
        # # decoding layers
        # self.up1 = UNetUp(512, 512)  # 256 256
        # self.up2 = UNetUp(1024, 256)  # 512 256
        # self.up3 = UNetUp(512, 128)  # 512 128
        # self.up4 = UNetUp(256, 64)  # 256 32
        # self.final = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, output_nc, 4, padding=1),
        #     nn.Tanh(),
        # )

    def forward(self,  input):
        """Standard forward"""
        x = self.model(input)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c = self.cen(c6)
        u6 = self.up6(c, c6)
        u5 = self.up5(u6, c5)
        u4 = self.up4(u5, c4)
        u3 = self.up3(u4, c3)
        u2 = self.up2(u3, c2)
        u1 = self.up1(u2, c1)
        f1 = self.out1(u1)


        return x

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        # conv1 = [nn.ReflectionPad2d(3),
        #               nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
        #               norm_layer(ngf),
        #               nn.ReLU(True)]
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # self.conv1 = nn.Sequential(*conv1)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        #inputs = self.conv1(input)
        return self.model(input)

class FullGenerator(nn.Module):
    """Create a fullallnet generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(FullGenerator, self).__init__()

        #construct full net
        features = [nn.Conv2d(input_nc, ngf * 4, kernel_size=5, stride=1, padding=2),
                    norm_layer(ngf * 4),
                    nn.LeakyReLU(True)]
        features += [nn.Conv2d(ngf * 4, ngf * 2, kernel_size=5, stride=1, padding=2),
                     norm_layer(ngf * 2),
                     nn.LeakyReLU(True)]
        # features = [nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1),
        #          norm_layer(ngf),
        #          nn.LeakyReLU(True)]
        # features += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
        #           norm_layer(ngf * 2),
        #           nn.LeakyReLU(True)]
        n_sampling = 2
        for i in range(n_sampling):
            mult = 2 ** i
            features += [nn.Conv2d(int(ngf * 2 / mult), int(ngf / mult), kernel_size=3, stride=1, padding=1),
                         norm_layer(int(ngf / mult)),
                         nn.LeakyReLU(True)]

        features += [nn.Conv2d(ngf // 2, output_nc, kernel_size=1, stride=1, padding=0),nn.Tanh()]
        self.features = nn.Sequential(*features)

        # construct unet structure
        # unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        #
        # for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # # self.conv1 = nn.Sequential(*conv1)
        # self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        #inputs = self.conv1(input)
        x = self.features(input)

        return x


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, ngf=None, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)

        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class GeneratorUgan3(nn.Module):
    """ A 7-layer UNet-based generator as described in the paper
    """

    def __init__(self, in_channels=3, ngf=64, norm_layer=nn.BatchNorm2d, output_nc=3, feature_scale=4, is_deconv=True,
                 is_batchnorm=True):
        super(GeneratorUgan3, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 512, 512, 512]
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], is_batchnorm=False)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.conv6 = unetConv2(filters[4], filters[5], self.is_batchnorm)
        self.cen = nn.Sequential(nn.Conv2d(filters[5], filters[6], 4, 2, 1),
                                 nn.ReLU(inplace=True))

        # upsampling
        # self.up_concat7 = unetUp(filters[6], filters[5], self.is_deconv)
        self.up6 = unetUp(filters[6], filters[5], self.is_deconv)
        self.up5 = unetUp(1024, filters[4], self.is_deconv)
        self.up4 = unetUp(1024, filters[3], self.is_deconv)
        self.up3 = unetUp(1024, filters[2], self.is_deconv)
        self.up2 = unetUp(512, filters[1], self.is_deconv)
        self.up1 = unetUp(256, filters[0], self.is_deconv)
        #
        self.o1 =  nn.Sequential(nn.ConvTranspose2d(filters[1], output_nc, 4, 2, 1, bias=False),
                                    nn.Tanh())
        # self.out1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
        #                           nn.Conv2d(filters[1], output_nc, 3, padding=1, bias=False),
        #                           )
    def forward(self, inputs):
        c1 = self.conv1(inputs)  # 16*512*1024
            # maxpool1 = self.maxpool1(conv1)  # 16*256*512
        c2 = self.conv2(c1)  # 32*256*512
            # maxpool2 = self.maxpool2(conv2)  # 32*128*256
        c3 = self.conv3(c2)  # 64*128*256
            # maxpool3 = self.maxpool3(conv3)  # 64*64*128
        c4 = self.conv4(c3)  # 128*64*128
            # maxpool4 = self.maxpool4(conv4)  # 128*32*64
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c = self.cen(c6)
        u6 = self.up6(c, c6)
        u5 = self.up5(u6, c5)
            # center = self.center(maxpool4)  # 256*32*64
        u4 = self.up4(u5, c4)  # 128*64*128
        u3 = self.up3(u4, c3)  # 64*128*256
        u2 = self.up2(u3, c2)  # 32*256*512
        u1 = self.up1(u2, c1)  # 16*512*1024
        f1 = self.o1(u1)
            #x = self.model(f1)

            # d1 = self.outconv1(up1)  # 256
        return f1

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_size, out_size, 4, padding=1, bias=False),
            #nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class UNet(nn.Module):

    def __init__(self, in_channels=3, output_nc=3, feature_scale=4, is_deconv=False, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 512]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], output_nc, 3, padding=1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    # def dotProduct(self,seg,cls):
    #     B, N, H, W = seg.size()
    #     seg = seg.view(B, N, H * W)
    #     final = torch.einsum("ijk,ij->ijk", [seg, cls])
    #     final = final.view(B, N, H, W)
    #     return final

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return F.sigmoid(d1)
# class UNet_3alex(nn.Module):
#
#     def __init__(self, input_nc=3, output_nc=3, feature_scale=4, is_deconv=True, is_batchnorm=True):
#         super(UNet_3alex, self).__init__()
#         self.is_deconv = is_deconv
#         self.in_channels = input_nc
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale
#
#         filters = [64, 128, 256, 512, 512]  #64, 128, 256, 512, 1024
#
#         ## -------------Encoder--------------
#         self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
#         self.maxpool1 = nn.AvgPool2d(kernel_size=2)
#
#         self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
#         self.maxpool2 = nn.AvgPool2d(kernel_size=2)
#
#         self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
#         self.maxpool3 = nn.AvgPool2d(kernel_size=2)
#
#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.maxpool4 = nn.AvgPool2d(kernel_size=2)
#
#         self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
#
#         ## -------------Decoder--------------
#         self.CatChannels = filters[0]
#         self.CatBlocks = 5
#         self.UpChannels = self.CatChannels * self.CatBlocks
#
#         '''stage 1'''
#         self.deconv5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=4, stride=2, padding=1)
#         self.deNorm5 = nn.BatchNorm2d(filters[3])
#         self.derelu5 = nn.ReLU(inplace=True)
#
#         self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=4, stride=2, padding=1)
#         self.deNorm4 = nn.BatchNorm2d(filters[2])
#         self.derelu4 = nn.ReLU(inplace=True)
#
#         self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=4, stride=2, padding=1)
#         self.deNorm3 = nn.BatchNorm2d(filters[1])
#         self.derelu3 = nn.ReLU(inplace=True)
#
#         self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=4, stride=2, padding=1)
#         self.deNorm2 = nn.BatchNorm2d(filters[0])
#         self.derelu2 = nn.ReLU(inplace=True)
#
#         # h1->320*320, hd1->320*320, Concatenation
#         self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
#         self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)
#
#         # hd2->160*160, hd1->320*320, Upsample 2 times
#         self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14*14
#         self.hd2_UT_hd1_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
#         self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)
#
#         # hd3->80*80, hd1->320*320, Upsample 4 times
#         self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 14*14
#         self.hd3_UT_hd1_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
#         self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)
#
#         # hd4->40*40, hd1->320*320, Upsample 8 times
#         self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 14*14
#         self.hd4_UT_hd1_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
#         self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)
#
#
#
#         # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
#         self.deconv1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
#         self.deNorm1 = nn.BatchNorm2d(self.UpChannels)
#         self.derelu1 = nn.ReLU(inplace=True)
#         # output
#         #self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
#         self.outconv1 = nn.Conv2d(self.UpChannels, output_nc, 3, padding=1)
#
#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm2d):
#                 init_weights(m, init_type='kaiming')
#
#     def forward(self, inputs):
#         ## -------------Encoder-------------
#         h1 = self.conv1(inputs)  # h1->320*320*64
#
#         h2 = self.maxpool1(h1)
#         h2 = self.conv2(h2)  # h2->160*160*128
#
#         h3 = self.maxpool2(h2)
#         h3 = self.conv3(h3)  # h3->80*80*256
#
#         h4 = self.maxpool3(h3)
#         h4 = self.conv4(h4)  # h4->40*40*512
#
#         h5 = self.maxpool4(h4)
#         hd5 = self.conv5(h5)  # h5->20*20*1024
#
#         ## -------------Decoder-------------
#
#         d5 = self.deconv5(hd5)
#         n5 = self.deNorm5(d5)
#         r5 = self.derelu5(n5)
#
#         d4 = self.deconv4(r5)
#         n4 = self.deNorm4(d4)
#         r4 = self.derelu4(n4)
#
#         d3 = self.deconv3(r4)
#         n3 = self.deNorm3(d3)
#         r3 = self.derelu3(n3)
#
#         d2 = self.deconv2(r3)
#         n2 = self.deNorm2(d2)
#         r2 = self.derelu2(n2)
#
#
#         h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
#         hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(h2))))
#         hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(h3))))
#         hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(h4))))
#         #hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(h5))))
#         hd1 = self.derelu1(self.deNorm1(self.deconv1(
#             torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, r2), 1)))) # hd1->320*320*UpChannels
#
#         #d1 = self.outconv1(hd1)  # d1->320*320*n_classes
#         d1 = self.outconv1(hd1)
#
#         return F.sigmoid(d1)
class UNet3Plus(nn.Module):
    def __init__(self, input_nc, output_nc=3, bilinear=True, feature_scale=4,
                 is_deconv=True, is_batchnorm=True):
        super(UNet3Plus, self).__init__()
        self.input_nc = input_nc
        self.output_nc= output_nc
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.input_nc, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, output_nc, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return F.sigmoid(d1)




class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4  #4
        padw = 1
        model = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]


        model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        #sequence += [nn.Conv2d(ndf * nf_mult, 3, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        #input_1 = input.view(input.size(0), -1)
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)



