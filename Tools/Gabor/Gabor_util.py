from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd
from skimage import io
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.color import rgb2gray

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels)*2), dtype=np.double)
    idx=0
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        feats[idx] = filtered.mean()
        idx=idx+1
        feats[idx] = filtered.var()
        idx=idx+1
    return feats

def Gabor_features(im,test=0,rotat=90):


    # prepare filter bank kernels
    kernels = []
    cnt=0
    for theta in range(4):
        theta = theta / 4. * np.pi
        for gamma in np.arange(0,0.51,0.1):
            kernel = np.real(cv2.getGaborKernel((5,5), 30,theta, 60, gamma,0))
            kernels.append(kernel)
            cnt=cnt+1
    im=rgb2gray(im)
    if test:
        im=nd.rotate(im, angle=rotat, reshape=False)
    im = img_as_float(im)

    return compute_feats(im, kernels)