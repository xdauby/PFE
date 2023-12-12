import numpy as np
import matplotlib.pyplot as plt


class TotalVariation:

    def __init__(self, weight='standard', sign='-'):

        self.weight = weight
        self.sign = sign


    def transform(self, x):

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
        
        return Dx, omega

    def transposed_transform(self, y):

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

        return im, omega
    