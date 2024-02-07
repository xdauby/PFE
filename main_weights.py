import os
import s3fs

import scipy.io
import astra
import matplotlib.pyplot as plt
import numpy as np
import astra.creators

from modules.algorithm.ChambollePockWeights import ChambollePockWeights
from modules.algorithm.ConjugateGradient import ConjugateGradient
from modules.operators.TotalVariation import TotalVariation
from modules.operators.Projection import Projection


phantom = scipy.io.loadmat('data/XCAT2D_PETCT.mat')
xtrue = phantom['mu_120']

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
xtrue = x_true_test[55]

# setup Astra projector
N = xtrue.shape[0]
pixel_size = 1
angles = np.linspace(0, np.pi, num=52)
detector_width = 1 

vol_geom = astra.creators.create_vol_geom(N, N)
proj_geom = astra.creators.create_proj_geom('parallel', detector_width, N, angles)
proj_id = astra.creators.create_projector('linear', proj_geom, vol_geom)

# generating projection
sinogram_id, b = astra.creators.create_sino(xtrue, proj_id)

#plt.imshow(b, cmap='gray')
#plt.pause(100)
print(b.shape)
plt.imsave('sino_astra.png', b, cmap='gray')
np.save('sino_astra.npy', b)


#plt.imsave('sinogram.png', b, cmap='gray')


# params for ChambollePock Algorithm, solving : || Ax - b ||^2_2 + || Hx ||_1

# tv.transform represents H and projection.transposed_transform represents HT
tv = TotalVariation()
# projection.transform represents A and projection.transposed_transform represents AT
projection = Projection(proj_id)
max_iter = 1200
max_inner_iter = 3
beta = 3e2
theta = 1
L = tv.norm(N,N)
sigma = 0.99 * (1e8 / (np.sqrt(1e8 * 1) * L))
tau = 0.99 * (1 / (np.sqrt(1e8 * 1) * L))

I = 3e5
bckg = 0 

ybar = I*np.exp(-b) + bckg
y = np.random.poisson(ybar)
#plt.imsave('noisy_sino_astra.png', y, cmap='gray')


#plt.imsave('noisy_sinogram.png', y, cmap='gray')

# initialize ChambollePock Algorithm
chambolle_pock = ChambollePockWeights(dim_image=N, 
                                max_iter=max_iter, 
                                max_inner_iter=max_inner_iter, 
                                beta=beta, 
                                theta=theta,
                                sigma=sigma, 
                                tau=tau, 
                                penalty_operator=tv)

# solve Ax = b with ChambollePock Algorithm
img = chambolle_pock.solve(y, I, bckg, projection)


plt.imsave('final_recon.png', img, cmap='gray')



