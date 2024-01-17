
import torch
from torch import nn
from modules.algorithm.ConjugateGradient import ConjugateGradient
from modules.operators.Wavelet import Wavelet
from modules.operators.RadonTorch import RadonTorch
from modules.operators.Operator import Operator
import torch.nn.functional as F
import numpy as np

class Conv2dPReLU(nn.Module):

    def __init__(self, in_chan=1, out_chan=64):
        super(Conv2dPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, padding=1, bias=True)
        self.prelu = nn.PReLU(out_chan, init=0.025)

    def forward(self, x):
        x = self.prelu(self.conv(x))
        return x


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, in_chan=1, out_chan=1,add_bias=True):
        super(DnCNN, self).__init__()
        layers = [Conv2dPReLU(in_chan=in_chan, out_chan=n_channels)]
        for _ in range(depth-2):
            layers.append(Conv2dPReLU(in_chan=n_channels, out_chan=n_channels))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_chan, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dncnn(x)
        return x

class MetaInvLOneIter(nn.Module):
    # One iteration of the HQS-CG algorithm
    def __init__(self):
        super(MetaInvLOneIter, self).__init__()
        self.CG = ConjugateGradient(10)
        self.CGInitDnCNN = DnCNN(depth=6, n_channels=8, in_chan=1, out_chan=1, add_bias=True)


    def forward(self, Y, xk_L, zk, lam_over_gamma , gamma, projection, wavelet):
    
        xkp1_0 = xk_L + self.CGInitDnCNN(xk_L)
        AtY = projection.transposed_transform(Y)

        gammaWtzk = gamma * wavelet.transposed_transform(zk)
        AtAxkp1_0 = projection.transposed_transform(projection.transform(xkp1_0))
        Wtwxkp1_0 = wavelet.transposed_transform(wavelet.transform(xkp1_0))
        residual = (AtAxkp1_0 + Wtwxkp1_0) - (AtY + gammaWtzk)
        # conjugate gradient
        xkp1_L = self.CG.solve(residual, xkp1_0, gamma, projection, wavelet)

        # soft tresholding
        Wxkp1_L = wavelet.transform(xkp1_L)
        zkp1 = F.relu(Wxkp1_L-lam_over_gamma) - F.relu(-Wxkp1_L-lam_over_gamma)

        return xkp1_L, zkp1



class MetaInvNetL(nn.Module):
    def __init__(self, layers, radon : Operator, wavelet : Operator):
        super(MetaInvNetL,self).__init__()
        self.layers = layers
        self.radon = radon
        self.wavelet = wavelet
        self.unrolled_net = nn.ModuleList()
        for i in range(self.layers + 1):
            self.unrolled_net.append(MetaInvLOneIter())
        

    def forward(self, y, xfbp):
        
        xk_list = [None] * (self.layers + 1)
        zk_list = [None] * (self.layers + 1)

        lambak = 0.005
        gammak = 0.01
        delta_lambda = 0.0008
        delta_gamma = 0.02
        
        z0 = self.wavelet.transform(xfbp)
        xk_list[0], zk_list[0] = self.unrolled_net[0](y, xfbp, z0, lambak, gammak, self.radon, self.wavelet)

        for i in range(self.layers):
            lambak -= delta_lambda
            gammak -= delta_gamma
            xk_list[i+1], zk_list[i+1] = self.unrolled_net[i+1](y, xk_list[i], zk_list[i], lambak, gammak, self.radon, self.wavelet)
      
        return xk_list