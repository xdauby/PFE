
import numpy as np
from modules.algorithm.IterativeAlgorithm import IterativeAlgorithm
from modules.operators.Operator import Operator
import matplotlib.pyplot as plt


class ConjugateGradient(IterativeAlgorithm):
    def __init__(self, max_iter : int) -> None:
        super().__init__(max_iter = max_iter)

    def initialize(self, residual, xk, mu, projection : Operator, wavelet : Operator) -> None: # wavelet : Operator

        self.A = lambda x: projection.transform(x)
        self.AT = lambda y: projection.transposed_transform(y)
        self.W = lambda x: wavelet.transform(x)
        self.WT = lambda y: wavelet.transposed_transform(y)

        self.rk = residual
        self.pk = residual
        self.xk = xk
        self.alphak = 0
        self.betak = 0
        self.mu = mu


    def one_step(self) -> None:


        rkTrk = np.linalg.norm(self.rk)**2
        #compute Gpk where G = (ATA + muWtW) (save results for future operations)
        Gpk = self.AT(self.A(self.pk)) + self.mu*self.WT(self.W(self.pk))

        #compute pkGpk
        pkGpk = np.sum(self.pk * Gpk)

        self.alphak = rkTrk/pkGpk
        self.xk = self.xk + self.alphak*self.pk
        self.rk = self.rk - self.alphak*Gpk
        self.betak = np.linalg.norm(self.rk)**2 / rkTrk
        self.pk = self.rk + self.betak * self.pk


    def solve(self, residual, xk, projection : Operator) -> np.array:
        self.initialize(residual, xk, projection)
        while self.iterations < self.max_iter:
            self.one_step()
            self.display(self.get_result(), self.iterations)
            self.iterations += 1
        return self.get_result()

    def get_result(self) -> np.array:
        return self.xk

    def display(self, x : np.array, i : int) -> None:
        plt.imshow(x, cmap='gray')
        plt.axis('off')
        plt.title(f'TV CP, iteration {i}')
        plt.pause(0.1)

# cg = ConjugateGradient(100)
# cg.solve(projection.transposed_transform(b), np.zeros((N, N)) , projection)
