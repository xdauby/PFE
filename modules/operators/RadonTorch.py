import numpy as np
from modules.operators.Operator import Operator
import torch
import torch_radon as tr

class RadonTorch(Operator):

    def __init__(self, n_rays, angles, volume) -> None:
        self.radon = tr.ParallelBeam(det_count = n_rays, 
                                     angles=angles, 
                                     volume=volume)

    def transform(self, x : torch.tensor) -> torch.tensor:
        Ax = self.radon.forward(x)
        return Ax

    def transposed_transform(self, y : torch.tensor) -> torch.tensor:
        ATy = self.radon.backward(y)
        return ATy

    def fbp(self, x : torch.tensor) -> torch.tensor:
        sinogram = self.radon.forward(x)
        filtered_sinogram = self.radon.filter_sinogram(sinogram)
        fbp = self.radon.backward(filtered_sinogram)
        return fbp

