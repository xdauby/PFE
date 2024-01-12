
import numpy as np
import matplotlib.pyplot as plt
from modules.algorithm.IterativeAlgorithm import IterativeAlgorithm
from modules.operators.Operator import Operator


class ChambollePock(IterativeAlgorithm):

    def __init__(self, 
                 dim_image : int, 
                 max_iter : int, 
                 max_inner_iter : int, 
                 beta : float, 
                 theta : float, 
                 sigma : float, 
                 tau : float, 
                 penalty_operator : Operator) -> None:
        
        super().__init__(max_iter = max_iter)

        self.dim_image = dim_image
        self.max_inner_iter = max_inner_iter
        self.beta = beta
        self.theta = theta
        self.sigma = sigma
        self.tau = tau
        self.penalty_operator = penalty_operator

    def initialize(self, b : np.array, projection : Operator) -> None:
        
        self.H = lambda x: self.penalty_operator.transform(x)
        self.HT = lambda y: self.penalty_operator.transposed_transform(y)
        self.A = lambda x: projection.transform(x)
        self.AT = lambda y: projection.transposed_transform(y)
        self.b = b

        self.x = np.zeros((self.dim_image, self.dim_image))
        self.xbar = self.x.copy()
        self.z = np.zeros_like(self.H(self.xbar))
        self.D_rec = self.AT(self.A(np.ones((self.dim_image, self.dim_image))))

    def one_step(self):

        # compute first proximal prox_sigma_h*
        z_temp = self.z.copy() + self.sigma * self.H(self.xbar.copy())
        self.z = np.sign(z_temp) * np.minimum(np.abs(z_temp), self.beta)
        
        # compute second proximal prox_tau_g
        x_old = self.x.copy()
        x_temp = x_old - self.tau * self.HT(self.z.copy())
        
        for _ in range(self.max_inner_iter):
            x_rec = self.x.copy() - self.AT(self.A(self.x.copy()) - self.b) / self.D_rec
            self.x = (self.D_rec * x_rec + x_temp / self.tau) / (self.D_rec + 1 / self.tau)
            self.x = np.maximum(self.x.copy(), 0)

        # extrapolation
        self.xbar = self.x + self.theta * (self.x - x_old)

    def get_result(self) -> np.array:
        return self.x
    
    def solve(self, b : np.array, projection : Operator) -> None:
        self.initialize(b, projection)
        while self.iterations < self.max_iter:
            self.one_step()
            self.display(self.get_result(), self.iterations)
            self.iterations += 1
        return self.get_result()

    def display(self, x : np.array, i : int) -> None:
        plt.imshow(x, cmap='gray')
        plt.axis('off')
        plt.title(f'TV CP, iteration {i}')
        plt.pause(0.1)

