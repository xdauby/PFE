import numpy as np
import matplotlib.pyplot as plt
from modules.operators.Operator import Operator
from typing import Tuple

class TotalVariation(Operator):

    def __init__(self, weight : str = 'standard', 
                 sign : str = '-', 
                 max_norm_iter : int = 100) -> None:

        self.weight = weight
        self.sign = sign
        self.max_norm_iter = max_norm_iter

    def transform(self, x : np.array) -> np.array:

        list_vals = [0, 1, -1]
        vect = np.array([[i, j] for i in list_vals for j in list_vals if not (i == 0 and j == 0)])
        
        if self.sign == '-':
            factor = -1

        Dx = np.zeros((x.shape[0], x.shape[1], 8))
        omega = np.zeros((vect.shape[0], 1))

        for i in range(vect.shape[0]):
            shift_vect = vect[i]
            omega[i] = 1 / np.linalg.norm(shift_vect)
            if self.weight == 'standard':
                Dx[:, :, i] = omega[i] * (x + factor * np.roll(np.roll(x, shift_vect[0], axis=0), shift_vect[1], axis=1))
        
        return Dx

    def transposed_transform(self, y : np.array) -> np.array:

        list_vals = [0, 1, -1]
        vect = np.array([[i, j] for i in list_vals for j in list_vals if not (i == 0 and j == 0)])
        
        if self.sign == '-':
            factor = -1

        im = np.zeros((y.shape[0], y.shape[1]))
        omega = np.zeros((vect.shape[0], 1))

        for i in range(vect.shape[0]):
            shift_vect = vect[i]
            im_shifted = np.roll(np.roll(y[:, :, i], -shift_vect[0], axis=0), -shift_vect[1], axis=1)
            omega[i] = 1 / np.linalg.norm(shift_vect)
            if self.weight == 'standard':
                im += omega[i] * (y[:, :, i] + factor * im_shifted)

        return im
    
    def norm(self, M : int, N : int) -> float:
        x, s = np.random.rand(M, N), 0
        for _ in range(self.max_norm_iter):
            x, s = self.norm_one_step(x, s)
        return s

    def norm_one_step(self, x : np.array, s : float) -> Tuple[np.array, float]:
        x = self.transposed_transform(self.transform(x))
        x = x / np.sqrt(np.sum(x**2))
        s = np.sqrt(np.sum(np.sum(np.sum(self.transform(x)**2))))
        return x, s