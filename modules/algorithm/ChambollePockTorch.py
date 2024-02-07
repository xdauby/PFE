import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from modules.algorithm.IterativeAlgorithm import IterativeAlgorithm
from modules.operators.Operator import Operator


class ChambollePockTorch(IterativeAlgorithm):

    def __init__(self, 
                 dim_image : int, 
                 max_iter : int, 
                 max_inner_iter : int, 
                 beta : float, 
                 theta : float, 
                 sigma : float, 
                 tau : float, 
                 penalty_operator : Operator,
                 device : str) -> None:
        
        super().__init__(max_iter = max_iter)

        self.dim_image = dim_image
        self.max_inner_iter = max_inner_iter
        self.beta = beta
        self.theta = theta
        self.sigma = sigma
        self.tau = tau
        self.penalty_operator = penalty_operator
        self.device = device

    def initialize(self, y : torch.Tensor, I : float, bckg : float, projection : Operator) -> None:
        
        self.H = lambda x: self.penalty_operator.transform(x)
        self.HT = lambda y: self.penalty_operator.transposed_transform(y)
        self.A = lambda x: projection.transform(x)
        self.AT = lambda y: projection.transposed_transform(y)
        self.b = y

        self.b = torch.log(I / (y - bckg)).to(self.device)
        self.b[y - bckg <= 0] = 0
        self.w = ((y - bckg)**2) / y
        self.w[y - bckg <= 0] = 0

        self.x = torch.zeros((1, 1, self.dim_image, self.dim_image)).to(self.device)
        self.xbar = self.x.clone()
        self.z = torch.zeros_like(self.H(self.xbar)).to(self.device)
        self.D_rec = self.AT(self.w*(self.A(torch.ones((1, 1, self.dim_image, self.dim_image)).to(self.device))))

    def one_step(self):

        # compute first proximal prox_sigma_h*
        z_temp = self.z + self.sigma * self.H(self.xbar)
        self.z = torch.sign(z_temp) * torch.minimum(torch.abs(z_temp), torch.tensor(self.beta).to(self.device))
        
        # compute second proximal prox_tau_g
        x_old = self.x.clone()
        x_temp = x_old - self.tau * self.HT(self.z)
        
        for _ in range(self.max_inner_iter):
            x_rec = self.x - self.AT(self.w*(self.A(self.x) - self.b)) / self.D_rec
            self.x = (self.D_rec * x_rec + x_temp / self.tau) / (self.D_rec + 1 / self.tau)
            self.x = torch.maximum(self.x, torch.tensor(0))

        # extrapolation
        self.xbar = self.x + self.theta * (self.x - x_old)

    def get_result(self) -> torch.Tensor:
        return self.x
    
    def solve(self, y : torch.Tensor, I : float, bckg : float, projection : Operator, display : bool = False) -> None:
        self.initialize(y, I, bckg, projection)
        while self.iterations < self.max_iter:
            self.one_step()
            if display:
            	self.display(self.get_result(), self.iterations)
            self.iterations += 1
        return self.get_result()

    def display(self, x : torch.Tensor, i : int) -> None:
        plt.imshow(TF.to_pil_image(x.squeeze()), cmap='gray')
        plt.axis('off')
        plt.title(f'TV CP, iteration {i}')
        plt.pause(0.1)

