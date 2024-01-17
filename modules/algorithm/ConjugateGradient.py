
import numpy as np
from modules.algorithm.IterativeAlgorithm import IterativeAlgorithm
from modules.operators.Operator import Operator
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class ConjugateGradient(IterativeAlgorithm):
    def __init__(self, max_iter : int) -> None:
        super().__init__(max_iter = max_iter)

    def initialize(self, residual, xk, lam, projection : Operator, wavelet : Operator) -> None: # wavelet : Operator

        self.A = lambda x: projection.transform(x)
        self.AT = lambda y: projection.transposed_transform(y)
        self.W = lambda x: wavelet.transform(x)
        self.WT = lambda y: wavelet.transposed_transform(y)

        self.rk = residual
        self.pk = residual
        self.xk = xk
        self.alphak = 0
        self.betak = 0
        self.lam = lam


    def one_step(self) -> None:


        rkTrk = torch.sum(self.rk**2,dim=(1,2,3))
        #compute Gpk where G = (ATA + lambda*WtW) (save results for future operations)
        Gpk = self.AT(self.A(self.pk)) + self.lam*self.WT(self.W(self.pk))

        #compute pkGpk
        pkGpk = torch.sum(self.pk * Gpk, dim=(1,2,3))

        self.alphak = rkTrk/pkGpk
        self.alphak = self.alphak.view(-1,1,1,1)
        self.xk = self.xk + self.alphak*self.pk
        self.rk = self.rk - self.alphak*Gpk
        self.betak = torch.sum(self.rk**2,dim=(1,2,3)) / rkTrk
        self.betak =self.betak.view(-1,1,1,1)
        self.pk = self.rk + self.betak * self.pk


    def solve(self, residual, xk, lam, projection, wavelet):
        self.initialize(residual, xk, lam, projection, wavelet)
        while self.iterations < self.max_iter:
            self.one_step()
            # self.display(self.get_result(), self.iterations)
            self.iterations += 1
        return self.get_result()

    def get_result(self) -> np.array:
        return self.xk

    def display(self, x , i : int) -> None:
        x = x.clone().detach().to('cpu')
        x = x.squeeze(0).squeeze(0)
        plt.imshow(x, cmap='gray')
        plt.axis('off')
        plt.title(f'TV CP, iteration {i}')
        plt.pause(0.1)

# cg = ConjugateGradient(100)
# cg.solve(projection.transposed_transform(b), np.zeros((N, N)) , projection)