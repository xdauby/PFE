import scipy.io
import astra
import matplotlib.pyplot as plt
import numpy as np
import astra.creators
from modules.algorithme.CP import CP
from modules.operators.TotalVariation import TotalVariation

phantom = scipy.io.loadmat('XCAT2D_PETCT.mat')

xtrue = phantom['mu_120']
N = xtrue.shape[0]
pixel_size = 1

angles = np.arange(0, np.pi + 0.001, 0.06)
detector_width = 1 ;

vol_geom = astra.creators.create_vol_geom(N, N);
proj_geom = astra.creators.create_proj_geom('parallel', detector_width, N, angles);
proj_id = astra.creators.create_projector('linear', proj_geom, vol_geom);

# generating the projection
sinogram_id, b = astra.creators.create_sino(xtrue,proj_id)

A = lambda x: astra.creators.create_sino(x, proj_id)[1]
AT = lambda y: astra.creators.create_backprojection(y, proj_id)[1]

tv = TotalVariation()

cp = CP(dim_image=N, 
        n_iter=100, 
        n_inner_iter=10, 
        n_norm_iter=100, 
        beta=0.5e-1, 
        theta=1,
        sigma=1e4, 
        tau=1, 
        operator=tv)

cp.descent(b, A, AT)






