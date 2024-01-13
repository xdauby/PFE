
import torch
from torch import nn
from modules.algorithm.ConjugateGradient import ConjugateGradient


class Conv2dPReLU(nn.Module):

    def __init__(self, in_chan=1, out_chan=64):
        super(InitCG, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, padding=1, bias=True))
        self.prelu = nn.PReLU(out_chan, init=0.025))

    def forward(self, x):
        x = self.prelu(self.conv(x))
        return x


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, in_chan=1, out_chan=1,add_bias=True):
        super(DnCNN, self).__init__()
        layers = [Conv2dPReLU()]
        for _ in range(depth-2):
            layers.append(Conv2dPReLU(in_channels=n_channels))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_chan, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dncnn(x)
        return x

class MetaInvL(nn.Module):
    '''MetaInvNet with light weight CG-Init'''
    def __init__(self):
        super(MetaInvL, self).__init__()
        self.CGInitDnCNN = DnCNN(depth=6, n_channels=8, in_chan=1, out_chan=1, add_bias=True)
        self.CG = ConjugateGradient(10)

    def forward(self, wavelet, sino, x, lam_over_mu, miu):
        Wu=wavelet.W(x)
        tresholding=nn.ReLU(Wu-lam_over_mu)- nn.ReLU(-Wu-lam_over_mu)
       
        AtY=CgModule.BwAt(sino)
        muWtV=CgModule.Wt(dnz)*miu

        uk0=x+self.CGInitDnCNN(x)
        Gx0=CgModule.AWx(uk0,miu)
        residual=Gx0-(AtY+muWtV)
        db=self.CG(CgModule, uk0, miu, res)

        return db,uk0

