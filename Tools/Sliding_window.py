from PIL import Image
import cv2
import numpy as np

def sliding_window(im, padding_size, window_size, window_shift):

    pad_x, pad_y= padding_size
    im_x, im_y, _ = im.shape
    zero_x = np.zeros((pad_x, im_y,3))
    im = np.concatenate((zero_x, im, zero_x), axis=0)
    im_x, im_y, _ = im.shape
    zero_y = np.zeros((im_x, pad_y, 3))
    im = np.concatenate((zero_y, im, zero_y), axis=1)
    im_x, im_y, _ = im.shape
    x_size, y_size = window_size
    x_shift, y_shift = window_shift 
    ls = []

    for loc_x in range(0, im_x-x_size+1, x_shift):
        for loc_y in range(0, im_y-y_size+1, y_shift):
            ls.append(im[loc_x:loc_x+x_size, loc_y:loc_y+y_size,:]) 
    return np.array(ls)

def Slide_Window(image2,im_size=256,block_num=4,padding=(0,0),window_shift_per=0.5):

    window_size = (int(im_size/block_num),int(im_size/block_num))
    window_shift=(int(window_size[0]*window_shift_per),int(window_size[1]*window_shift_per))
    
    if im_size/block_num != int(im_size/block_num):
        print("Warning! the number of blocks is NOT integer")
    
    slid = sliding_window(image2, padding, window_size, window_shift)
    return slid
