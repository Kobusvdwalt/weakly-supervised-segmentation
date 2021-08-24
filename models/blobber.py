import torch
import numpy as np
from torch import tensor

def gaus_kernel(shape=(3,3),sigma=10):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

class Blobber(torch.nn.Module):
    def __init__(self, kernel_size = 3, iterations = 1):
        super().__init__()

        self.iterations = iterations
        self.kernel_size = kernel_size

        kernel = gaus_kernel(shape=(kernel_size, kernel_size))

        conv = torch.nn.Conv2d(1, 1, kernel_size, padding=(kernel_size-1)//2)
        conv.bias.data.fill_(0)
        conv.weight.data.copy_(torch.tensor(kernel))

        self.blob_conv = conv
        self.blob_sigm = torch.nn.Sigmoid()

        for p in self.blob_conv.parameters():
            p.requires_grad = False

    # def expand(self, inputs):
    #     blob = inputs
    #     blob = self.blob_conv(blob)
    #     blob = self.blob_sigm((blob - 0.1) * 100)
    #     return blob
    
    # def shrink(self, inputs):
    #     blob = inputs
    #     blob = self.blob_conv(blob)
    #     blob = self.blob_sigm((blob - 0.9) * 100)
    #     return blob

    def forward(self, inputs):
        # blob = self.expand(inputs)
        # blob = self.shrink(blob)
        # blob[blob>0.5] = 1
        # blob[blob<=0.5] = 0
        return blob