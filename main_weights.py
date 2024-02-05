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

#plt.imsave('sinogram.png', b, cmap='gray')


# params for ChambollePock Algorithm, solving : || Ax - b ||^2_2 + || Hx ||_1

# tv.transform represents H and projection.transposed_transform represents HT
tv = TotalVariation()
# projection.transform represents A and projection.transposed_transform represents AT
projection = Projection(proj_id)
max_iter = 100
max_inner_iter = 10
beta = 0.5e-1
theta = 1
L = tv.norm(N,N)
sigma = 0.99 * (1e4 / (np.sqrt(1e4 * 1) * L))
tau = 0.99 * (1 / (np.sqrt(1e4 * 1) * L))

print("L:", L)

I = 3e5
bckg = 0 

ybar = I*np.exp(-b) + bckg ;
y = np.random.poisson(ybar) ;

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



