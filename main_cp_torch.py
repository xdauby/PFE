"""
main_cp_torch.py
"""
import os
import s3fs

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_radon as tr

from modules.algorithm.ChambollePockTorch import ChambollePockTorch
from modules.operators.TotalVariationTorch import TotalVariationTorch
from modules.operators.RadonTorch import RadonTorch

# set device (has to be 'cuda' since torch_radon need gpu support
device = 'cuda'

# connection to database
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

BUCKET_X_TRUE_TEST = "clemphg/x_true_test"
x_true_test_filenames = fs.ls(BUCKET_X_TRUE_TEST)[1:]

# load data
def import_data(file_paths):
    data = []
    for file_path in file_paths:
        with fs.open(file_path, mode="rb") as file_in:
            data.append(np.load(file_in, encoding="bytes"))
    return data
    
x_true_test = import_data(x_true_test_filenames)

# select image 
xtrue = x_true_test_filenames[55]

# transform it into 4D tensor
xtrue_tensor = torch.from_numpy(np.array([xtrue])).to(dtype=torch.float32).to(device).unsqueeze(1)
#plt.imsave('xtrue.png', xtrue_tensor.cpu().numpy(), cmap='gray')

# setup projector
N = xtrue.shape[0]

angles = np.linspace(0, np.pi, 52, endpoint=False)
volume = tr.Volume2D()
volume.set_size(height=N, width=N)
radon = RadonTorch(N, angles, volume)

# create sinogram
sino_xtrue_tensor = radon.transform(xtrue_tensor)
#print(xtrue_tensor.shape)
#plt.imsave('sinogram.png', sino_xtrue_tensor.squeeze().cpu().numpy(), cmap='gray')

# parameters for the reconstruction
tv = TotalVariationTorch()

max_iter = 1200
max_inner_iter = 3
beta = 3e2
theta = 1
L = tv.norm(N,N)
sigma = 0.99 * (1e8 / (np.sqrt(1e8 * 1) * L))
tau = 0.99 * (1 / (np.sqrt(1e8 * 1) * L))

I = 3e5
bckg = 0 

# noisy sinogram
y = torch.poisson(I*torch.exp(-sino_xtrue_tensor) + bckg)

#print(y.shape)
#plt.imsave('noisy_sinogram.png', y.squeeze().cpu().numpy(), cmap='gray')

# initialize ChambollePock Algorithm
chambolle_pock = ChambollePockTorch(dim_image=N, 
									max_iter=max_iter, 
                                	max_inner_iter=max_inner_iter, 
                                	beta=beta, 
                                	theta=theta,
                                	sigma=sigma, 
                                	tau=tau, 
                                	penalty_operator=tv,
                                    device=device)

start_time = time.time()

# solve Ax = b with ChambollePock Algorithm
img = chambolle_pock.solve(y, I, bckg, radon)

print("Time:", time.time()-start_time)

plt.imsave('final_recon.png', img.squeeze().cpu().numpy(), cmap='gray')

