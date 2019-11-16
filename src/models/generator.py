import torch
from torch.nn import Module, Linear, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import tanh, interpolate, sigmoid
from .spade_resblk import SPADEResBlk

class SPADEGenerator(Module):
    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.gen_input_size, args.gen_hidden_size)
        self.spade_resblk1 = SPADEResBlk(args, 128)
        self.conv1 = Conv2d(128, 64, 3, padding=1)
        self.spade_resblk2 = SPADEResBlk(args, 64)
        self.conv2 = Conv2d(64, 32, 3, padding=1)
        self.spade_resblk3 = SPADEResBlk(args, 32)
        self.conv3 = Conv2d(32, 16, 3, padding=1)
        self.spade_resblk4 = SPADEResBlk(args, 16)
        self.conv4 = Conv2d(16, 8, 3, padding=1)
        self.spade_resblk5 = SPADEResBlk(args, 8)
        self.conv = spectral_norm(Conv2d(8, 3, kernel_size=(3,3), padding=1))

    def forward(self, x, seg):
        b, c, h, w = seg.size()
        x = self.linear(x)
        x = x.view(b, -1, 4, 4)

        seg2 = interpolate(seg, size=(4, 4))
        x = interpolate(self.spade_resblk1(x, seg2), size=(16, 16))
        x = self.conv1(x)
        seg2 = interpolate(seg, size=(16, 16))
        x = interpolate(self.spade_resblk2(x, seg2), size=(32, 32))
        seg2 = interpolate(seg, size=(32, 32))
        x = self.conv2(x)
        x = interpolate(self.spade_resblk3(x, seg2), size=(64, 64))
        seg2 = interpolate(seg, size=(64, 64))
        x = self.conv3(x)
        x = interpolate(self.spade_resblk4(x, seg2), size=(128, 128))
        seg2 = interpolate(seg, size=(128, 128))
        x = self.conv4(x)
        x = interpolate(self.spade_resblk5(x, seg2), size=(256, 256))
        x = tanh(self.conv(x))
        return x