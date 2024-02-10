import torch
from pytorch_msssim import SSIM, MS_SSIM
import torch.nn as nn
import torch.nn.functional as F


def metainvnet_loss_l1(pred, y):
    layer = len(pred)
    loss = 0.0
    l1loss = torch.nn.L1Loss()
    for l in range(layer):
        loss += l1loss(pred[l], y) * (1.1**l)
    return loss

def metainvnet_loss_l2(pred, y):
    layer = len(pred)
    loss = 0.0
    l2loss = torch.nn.MSELoss()
    for l in range(layer):
        loss += l2loss(pred[l], y) * (1.1**l)
    return loss

def metainvnet_ms_ssim_loss(pred, y):
    layer = len(pred)
    loss = 0.0
    for l in range(layer):
        data_range = torch.max(torch.tensor([torch.max(pred[l]) - torch.min(pred[l]), torch.max(y) - torch.min(y)]))
        ms_ssim = MS_SSIM(win_size=11, win_sigma=1.5, data_range=data_range, size_average=True, channel=1)
        loss += (1 - ms_ssim(pred[l], y)) * (1.1**l)
    return loss


class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        lM = l[:, -1, :, :] #* l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]
        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels

        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-1, length=1),
                               groups=1, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()


def ms_ssim_l1_loss(pred, y):
    data_range = torch.max(torch.tensor([torch.max(pred) - torch.min(pred), torch.max(y) - torch.min(y)], requires_grad=False))
    loss_ms_ssim_l1 = MS_SSIM_L1_LOSS(data_range=data_range, alpha=0.15)
    loss = loss_ms_ssim_l1(pred, y)
    return loss



