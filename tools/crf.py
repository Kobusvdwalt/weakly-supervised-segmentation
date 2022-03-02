import cv2
import torch
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax

class CRFParams:
    gaus_sxy: 10
    bilateral_sxy: 100
    bilateral_rgb: 10

class CRF:
    def __init__(self):
        pass

    def process(self, image, mask):
        image = np.array(image, dtype=np.ubyte)
        image = np.copy(image, order='C')

        n_classes = mask.shape[0]
        mask = mask
        mask = torch.softmax(torch.tensor(mask), dim=0).cpu().numpy()

        unary = unary_from_softmax(mask)
        unary = np.ascontiguousarray(unary)

        denseCrf = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_classes)  # width, height, nlabels
        denseCrf.setUnaryEnergy(unary)

        xy_1 = 10
        xy_2 = 100
        rgb = 10

        denseCrf.addPairwiseGaussian(sxy=(xy_1, xy_1), compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        denseCrf.addPairwiseBilateral(sxy=(xy_2, xy_2), srgb=(rgb, rgb, rgb), rgbim=image,
                            compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        result = denseCrf.inference(5)
        result = np.array(result).reshape((21, image.shape[0], image.shape[1]))

        return result
