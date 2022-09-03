import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

def image_blocking(im_path,im_size=256,block_num=4,test_im=0,show_test=0):
    image=Image.open(im_path)
    image=np.asarray(image)
    image2=cv2.resize(image,[im_size,im_size])
    z=np.vsplit(image2, block_num)
    b=np.hsplit(z[0][:][:][:],block_num)
    b=np.asarray(b)
    c=b[np.newaxis,:]
    for i in range(1,block_num):
        b=np.hsplit(z[i][:][:][:],block_num)
        b=np.asarray(b)
        b=b[np.newaxis,:]
        c=np.append(c,b,axis=0)
    if test_im:
        rebuilt_img=np.hstack(c[0][:])
        for row in range(1,block_num):
            w=np.hstack(c[row][:])
            rebuilt_img=np.vstack([rebuilt_img,w])
        if show_test:
            fig = plt.figure(im_path)
            plt.suptitle(im_path)
            ax = fig.add_subplot(1, 2, 1,title='Rebuilt')
            plt.imshow(rebuilt_img)
            plt.axis("off")            
            ax = fig.add_subplot(1, 2, 2,title='Original')
            plt.imshow(image2)
            plt.axis("off")
            plt.show()


    return c  