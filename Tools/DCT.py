from . import Sliding_window as SW
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct
def rotation(image):
    imsize=np.shape(image)
    im=np.zeros((4,imsize[0],imsize[1],imsize[2]))
    im[0]=image
    image_90 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im[1]=image_90
    image_180 = cv2.rotate(image_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im[2]=image_180
    image_270 = cv2.rotate(image_180, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im[3]=image_270
    return im
    
def plt_imshow(image,title=''):

    image = cv2.cvtColor(np.array(image,dtype=np.uint16), cv2.COLOR_BGR2RGB) #The conversion for data type is because cv2 only accept a few datatypes
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

def dct2(a):
    return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def bin_dec(hashC):
    hashV=0
    len_code=len(hashC)-1
    for i in range(len_code+1):
        if hashC[i]=='1':
            hashV=hashV+2**(len_code-i)
    return hashV

def windows_comput(imp,crop=True):
    if crop:
        crop_col=64
        crop_col_w=128
        crop_row=64
        crop_row_h=128

    im_size=256
    block_num=4
    padding=(0,0)
    window_shift_per=0.5

    image1=Image.open(imp)
    image1=np.asarray(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2=cv2.resize(image1,[im_size,im_size])
    if crop:
        image_undertaken=image2[crop_row:crop_row+crop_row_h,crop_col:crop_col+crop_col_w]
    else:
        image_undertaken=image2
        
    imsize=np.shape(image_undertaken)
    image_rotated=rotation(image_undertaken)

    slid_window=[]

    for rot in range(np.shape(image_rotated)[0]):
        im=image_rotated[rot]
        im_s=np.shape(image_rotated[rot])[0]

        slid_window.append( SW.Slide_Window(im,im_size=im_s,block_num=block_num,padding=padding,window_shift_per=window_shift_per))
    return slid_window

def DCT_hash_compute(slid_window):
    num_R,num_W,row,col,ch=np.shape(slid_window)
    dct_avg=np.zeros((num_R,num_W))
    for rot in range(num_R):
        for win in range(num_W):
            image_to_dct=slid_window[rot][win]
            imsize = image_to_dct.shape
            dct1 = np.zeros(imsize)
            for c in range(imsize[2]):
                dct1[:,:,c] = dct2( image_to_dct[:,:,c] )
            dct_avg[rot,win]=np.average(dct1)

    dct_median=np.median(dct_avg)

    hash_code=""
    for rot in range(num_R):
        for win in range(num_W):
            if dct_avg[rot,win]<dct_median:
                hash_code=hash_code+"0"
            else:
                hash_code=hash_code+"1"
    return hash_code


