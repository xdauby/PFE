
import torch

class TotalVariationTorch:

    def __init__(self):
        pass
    
    def transform(self, x):
        
        x1 = (x - torch.roll(x, shifts=(0, 1), dims=(2, 3))) 
        x2 = (x - torch.roll(x, shifts=(0, -1), dims=(2, 3))) 
        x3 = (x - torch.roll(x, shifts=(1, 0), dims=(2, 3))) 
        x4 = (x - torch.roll(x, shifts=(-1, 0), dims=(2, 3))) 
        
        x5 = (1/sqrt(2)) * (x - torch.roll(x, shifts=(1, -1), dims=(2, 3))) 
        x6 = (1/sqrt(2)) * (x - torch.roll(x, shifts=(-1, 1), dims=(2, 3))) 
        x7 = (1/sqrt(2)) * (x - torch.roll(x, shifts=(1, 1), dims=(2, 3))) 
        x8 = (1/sqrt(2)) * (x - torch.roll(x, shifts=(-1, -1), dims=(2, 3))) 
        
        return torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
    
    def transposed_transform(self, x):
        
        x1 = (x[:,0:1,:,:] - torch.roll(x[:,0:1,:,:], shifts=(0, -1), dims=(2, 3)))
        x2 = (x[:,1:2,:,:] - torch.roll(x[:,1:2,:,:], shifts=(0, 1), dims=(2, 3)))
        x3 = (x[:,2:3,:,:] - torch.roll(x[:,2:3,:,:], shifts=(-1, 0), dims=(2, 3)))
        x4 = (x[:,3:4,:,:] - torch.roll(x[:,3:4,:,:], shifts=(1, 0), dims=(2, 3)))
        
        x5 = (1/sqrt(2)) * (x[:,4:5,:,:] - torch.roll(x[:,4:5,:,:], shifts=(-1, 1), dims=(2, 3)))
        x6 = (1/sqrt(2)) * (x[:,5:6,:,:] - torch.roll(x[:,5:6,:,:], shifts=(1, -1), dims=(2, 3)))
        x7 = (1/sqrt(2)) * (x[:,6:7,:,:] - torch.roll(x[:,6:7,:,:], shifts=(-1, -1), dims=(2, 3)))
        x8 = (1/sqrt(2)) * (x[:,7:8,:,:] - torch.roll(x[:,7:8,:,:], shifts=(1, 1), dims=(2, 3)))
        
        return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8