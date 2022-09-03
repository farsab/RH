from Tools import Sliding_window as SW
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt

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

def bin_dec(hashC):
    hashV=0
    len_code=len(hashC)-1
    # print(len_code)
    for i in range(len_code+1):
        if hashC[i]=='1':
            hashV=hashV+2**(len_code-i)
    return hashV

def windows_comput(imp,block_num=8):
    im_size=256
    padding=(0,0)
    window_shift_per=0.5

    image1=Image.open(imp)
    image1=np.asarray(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2=cv2.resize(image1,[im_size,im_size])
    image_undertaken=image2
        
    imsize=np.shape(image_undertaken)
    image_rotated=rotation(image_undertaken)

    slid_window=[]

    for rot in range(np.shape(image_rotated)[0]):
        im=image_rotated[rot]
        im_s=np.shape(image_rotated[rot])[0]

        slid_window.append( SW.Slide_Window(im,im_size=im_s,block_num=block_num,padding=padding,window_shift_per=window_shift_per))
    return slid_window

def DWT_hash_compute(slid_window):
    num_R,num_W,row,col,ch=np.shape(slid_window)
    dwt_avg=np.zeros((num_R,num_W))
    for rot in range(num_R):
        for win in range(num_W):
            image_to_dwt=slid_window[rot][win]
            
            imsize = list(image_to_dwt.shape)
            imsize[0]=int(imsize[0]/2)
            imsize[1]=int(imsize[1]/2)
            dwt1 = np.zeros(imsize)
            for c in range(imsize[2]):
                coef=dwt2d(image_to_dwt[:,:,c])
                LL, (LH, HL, HH) = coef
                dwt1[:,:,c] = LL
            dwt_avg[rot,win]=np.average(dwt1)

    dwt_median=np.median(dwt_avg)

    bin_code=""
    for rot in range(num_R):
        for win in range(num_W):
            if dwt_avg[rot,win]<dwt_median:
                bin_code=bin_code+"0"
            else:
                bin_code=bin_code+"1"
    return bin_code

def bin_two_parts(binCode):
    if len(binCode)%2==0:
        len_part_1=len_part_2=int(len(binCode)/2)
        if (len_part_1+len_part_2)!=len(binCode):
            raise Exception("Error!!!!: len_part1+len_part2 is not equal to length of binary code")
    else:
        len_part_1=int(len(binCode)/2)+1
        len_part_1=int(len(binCode)/2)

    binary_code_p1=binCode[:len_part_1]
    binary_code_p2=binCode[len_part_1:]
    
    return binary_code_p1,binary_code_p2


def dwt2d(im_dw):
    return pywt.dwt2(im_dw,'db1')

