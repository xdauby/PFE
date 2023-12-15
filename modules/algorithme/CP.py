
import numpy as np
import matplotlib.pyplot as plt

class CP:

    def __init__(self, dim_image, n_iter, n_inner_iter, n_norm_iter, beta, theta, sigma, tau, operator):
        
        self.dim_image = dim_image
        self.n_iter = n_iter
        self.n_inner_iter = n_inner_iter
        self.n_norm_iter = n_norm_iter
        self.beta = beta
        self.theta = theta
        self.sigma = sigma
        self.tau = tau

        self.operator = operator

    def norm_operator(self, x, s=0):
        for _ in range(self.n_norm_iter):
            x, s = self.norm_operator_one_iter(x, s)
        return s

    def norm_operator_one_iter(self, x, s):
        x = self.operator.transposed_transform(self.operator.transform(x)[0])[0]
        x = x / np.sqrt(np.sum(x**2))
        s = np.sqrt(np.sum(np.sum(np.sum(self.operator.transform(x)[0]**2))))
        return x, s

    def display(self, x, i):
        plt.imshow(x, cmap='gray')
        plt.axis('off')
        plt.title(f'TV CP, iteration {i}')
        plt.pause(0.1)


    def descent(self, b, A, AT, on_display=True):
        
        N = self.dim_image
        sigma = self.sigma

        H = lambda x: self.operator.transform(x)
        HT = lambda y: self.operator.transposed_transform(y)

        rand_x = np.random.rand(N, N)
        L = self.norm_operator(rand_x)

        sigmatau = self.sigma * self.tau
        self.sigma = self.sigma / (np.sqrt(sigmatau) * L)
        self.tau = self.tau / (np.sqrt(sigmatau) * L)

        self.sigma *= 0.99
        self.tau *= 0.99

        x = np.zeros((N, N))
        xbar = x
        z = np.zeros_like(H(xbar)[0])

        D_rec = AT(A(np.ones((N, N))))

        for i in range(self.n_iter):
            
            if on_display:
                self.display(x, i)


            # compute first proximal prox_sigma_h*
            z_temp = z + self.sigma * H(xbar)[0]
            z = np.sign(z_temp) * np.minimum(np.abs(z_temp), self.beta)
            
            # compute second proximal prox_tau_g
            x_old = x
            x_temp = x_old - self.tau * HT(z)[0]
            
            # Approximate minimum of 1/2 ||Ax - b||²_w + (1/2) * (1/tau) * ||x - x_temp||² 
            # by noting g(.|x_n) the SPS for 1/2 ||Ax - b||²_w and h(.|x_n) the SPS for (1/2)||x - x_temp||² 
            # we try to find the minimu of g(.|x_n) + (1/tau) * h(.|x_n)
            # where g(x|x_n) = (1/2) * ||x - ( (x_n) - (D_rer)^{-1} (AT(A(x_n) - b)) ||²_(D_rec)
            # and h(x|x_n) = (1/2) * ||x - ( (x_n) - (x_n - x_temp) ||²_(Id) = || x - x_temp ||²

            for _ in range(self.n_inner_iter):
                x_rec = x - AT(A(x) - b) / D_rec
                x = (D_rec * x_rec + x_temp / self.tau) / (D_rec + 1 / self.tau)
                x = np.maximum(x, 0)

            # extrapolation
            xbar = x + self.theta * (x - x_old)

        return x