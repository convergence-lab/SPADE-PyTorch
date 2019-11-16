import torch
from torch.nn import Module, Linear, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import tanh, interpolate, sigmoid
from .spade_resblk import SPADEResBlk

class SPADEGenerator(Module):
    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.gen_input_size, args.gen_hidden_size)
        self.spade_resblk1 = SPADEResBlk(args, 4096)
        self.spade_resblk2 = SPADEResBlk(args, 1024)
        self.spade_resblk3 = SPADEResBlk(args, 1024)
        self.spade_resblk4 = SPADEResBlk(args, 256)
        self.spade_resblk5 = SPADEResBlk(args, 64)
        self.spade_resblk6 = SPADEResBlk(args, 16)
        self.spade_resblk7 = SPADEResBlk(args, 4)
        self.conv = spectral_norm(Conv2d(1, 3, kernel_size=(3,3), padding=1))

    def forward(self, x, seg):
        b, c, h, w = seg.size()
        x = self.linear(x)
        x = x.view(b, -1, 4, 4)

        x = self.spade_resblk1(x, seg)
        x = x.reshape(b, 1024, 8, 8)
        x = self.spade_resblk2(x, seg)
        x = self.spade_resblk3(x, seg)
        x = x.reshape(b, 256, 16, 16)
        x = self.spade_resblk4(x, seg)
        x = x.reshape(b, 64, 32, 32)
        x = self.spade_resblk5(x, seg)
        x = x.reshape(b, 16, 64, 64)
        x = self.spade_resblk6(x, seg)
        x = x.reshape(b, 4, 128, 128)
        x = self.spade_resblk7(x, seg)        
        x = x.reshape(b, 1, 256, 256)
        x = tanh(self.conv(x))
        return x