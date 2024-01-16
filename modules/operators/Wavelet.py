import numpy as np
import torch 
from modules.operators.Operator import Operator
import torch.nn.functional as F


class Wavelet(Operator):

    def __init__(self) -> None:

        D1=np.array([1.0, 2, 1])/4
        D2=np.array([1, 0, -1])/4*np.sqrt(2)
        D3=np.array([-1 ,2 ,-1])/4
        D4='ccc'
        R1=np.array([1, 2, 1])/4
        R2=np.array([-1, 0, 1])/4*np.sqrt(2)
        R3=np.array([-1, 2 ,-1])/4
        R4='ccc'
        D=[D1,D2,D3,D4]
        R=[R1,R2,R3,R4]
        
        D_tmp=torch.zeros(3,1,3,1)
        for ll in range(3):
            D_tmp[ll,]=torch.from_numpy(np.reshape(D[ll],(-1,1)))

        W=D_tmp
        W2=W.permute(0,1,3,2)
        kernel_dec=np.kron(W.numpy(),W2.numpy())

        R_tmp=torch.zeros(3,1,1,3)
        for ll in range(3):
            R_tmp[ll,]=torch.from_numpy(np.reshape(R[ll],(1,-1)))

        R=R_tmp
        R2=R_tmp.permute(0,1,3,2)
        kernel_rec=np.kron(R2.numpy(),R.numpy())

        self.kernel_dec=torch.tensor(kernel_dec,requires_grad=False,dtype=torch.float32).cuda()
        self.kernel_rec=torch.tensor(kernel_rec,requires_grad=False,dtype=torch.float32).view(1,9,3,3).view(9,1,3,3).cuda()

    def transform(self, x : torch.Tensor) -> torch.Tensor:
        Wx=F.conv2d(F.pad(x, (1,1,1,1), mode='circular'), self.kernel_dec[1:,...])
        return Wx

    def transposed_transform(self, y : torch.Tensor) -> torch.Tensor:
        tem_coeff=F.conv2d(F.pad(y, (1,1,1,1), mode='circular'), self.kernel_rec[1:,:,...],groups=8)
        Wty=torch.sum(tem_coeff,dim=1,keepdim=True)
        return Wty

