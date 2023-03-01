import torch.nn as nn
import torch
import torch.nn.functional as F
from dehazemodels.ddf import DDFPack
# from ddf import DDFPack

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockDynamic_1x9(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockDynamic_1x9, self).__init__()
        self.res_scale = res_scale
        self.conv1 = DDFPack(in_channels=num_feat)
        self.conv2 = nn.Conv2d(num_feat, num_feat, (1, 9), 1, (0, 4), bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlockDynamic_3x3(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockDynamic_3x3, self).__init__()
        self.res_scale = res_scale
        self.conv1 = DDFPack(in_channels=num_feat)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, is_dynamic=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.is_dynamic = is_dynamic
        if self.is_dynamic:
            self.ddf = DDFPack(num_feat)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        identity = x
        if self.is_dynamic:
            x = self.lrelu(self.ddf(x)) + x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlockNoBN_1x9(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, is_dynamic=False):
        super(ResidualBlockNoBN_1x9, self).__init__()
        self.res_scale = res_scale
        self.is_dynamic = is_dynamic
        if self.is_dynamic:
            self.ddf = DDFPack(num_feat)
        self.conv1 = nn.Conv2d(num_feat, num_feat, (1, 9), 1, (0, 4), bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, (1, 9), 1, (0, 4), bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        identity = x
        if self.is_dynamic:
            x = self.lrelu(self.ddf(x)) + x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """
    def __init__(self, in_channels, out_channels=64, stride=1, num_blocks=30):
        super().__init__()
        main = []
        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, num_feat=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class ResidualBlocksWithEndDeConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """
    def __init__(self, in_channels, out_channels=64, stride=1, num_blocks=30):
        super().__init__()

        main = []
        # a convolution used to match the channels of the residual blocks
        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, num_feat=in_channels))
        main.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
        
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        c = input.shape[1]

        kernel = torch.zeros(size=[self.downscale_factor * self.downscale_factor * c,
                                1, self.downscale_factor, self.downscale_factor],
                            device=input.device)
        for y in range(self.downscale_factor):
            for x in range(self.downscale_factor):
                kernel[x + y * self.downscale_factor::self.downscale_factor*self.downscale_factor, 0, y, x] = 1
        return F.conv2d(input, kernel, stride=self.downscale_factor, groups=c)

if __name__ == '__main__':
    a = torch.rand(1, 32, 64, 64).cuda()
    net = ResidualBlockDynamic_3x3(32).cuda()
    b = net(a)
    print(b.shape)