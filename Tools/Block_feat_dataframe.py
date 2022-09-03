import pandas as pd
import numpy as np
from .Blocking import Block_util
from . import LLF
from . import HLF
from IPython.display import clear_output
import time


def Block_feat_dataframe_LLF(im_p,im_size=256,block_num=4,test_im=0,show_test=0):
        image_idx=0
        df = pd.DataFrame([], columns =[str(i) for i in range (16)]) 
        df.insert(0,"filename",[])
        df_error=[]
        for impath in im_p:
                print (impath)

                Blocked_image=Block_util.image_blocking(impath,im_size,block_num,test_im,show_test)
                idx=1
                Blocked_image_feature=np.zeros((np.shape(Blocked_image)[0],np.shape(Blocked_image)[0],156))
                for i in range(np.shape(Blocked_image)[0]):
                        for j in range(np.shape(Blocked_image)[1]):
                                print("\rCalculating Features for Block #",idx, end="")
                                Blocked_image_feature[i][j],g1err,g2err,nanerr=LLF.compute_all_feautres(Blocked_image[i][j])
                                idx=idx+1
                if (g1err!=0 or g2err!=0 or nanerr!=0):
                        np.append(df_error,impath,axis=0)
                print(f"       Image # {image_idx} is done!")
                bsize=np.shape(Blocked_image_feature)[0]
                

                idx=0
                df.at[image_idx,'filename']=impath
                for i in range(bsize):
                        for j in range(bsize):
                                df.at[image_idx,str(idx)]=Blocked_image_feature[i][j]
                                #print(f"{i}   {j}  {np.count_nonzero(Blocked_image_feature[i][j])}")
                                idx=idx+1
                image_idx=image_idx+1
        # clear_output(wait=True)
        # time.sleep(0.1)
        print ("No errors so far! You are an awesome programmer.")
        weight_shape=np.shape(Blocked_image_feature)[2]  # Weight dimension 156*156
        print (f"The feature set needs {Blocked_image_feature.nbytes} bytes")
        n_zeros = np.count_nonzero(Blocked_image_feature==0)
        print (f"There are {n_zeros} zeros out of {np.shape(Blocked_image)[0]**2*156}")

        return df,weight_shape,g1err,g2err,nanerr,df_error

def Block_feat_dataframe_HLF(model,im_p,im_size=256,block_num=4,test_im=0,show_test=0):
        #print("\n")
        image_idx=0
        df = pd.DataFrame([], columns =[str(i) for i in range (16)]) 
        df.insert(0,"filename",[])
        df_error=[]
        for impath in im_p:
                print (impath)

                Blocked_image=Block_util.image_blocking(impath,im_size,block_num,test_im,show_test)
                idx=1
                Blocked_image_feature=np.zeros((np.shape(Blocked_image)[0],np.shape(Blocked_image)[0],156))
                for i in range(np.shape(Blocked_image)[0]):
                        for j in range(np.shape(Blocked_image)[1]):
                                print("\rCalculating Features for Block #",idx, end="")
                                Blocked_image_feature[i][j]=HLF.compute_all_feautres_HLF(model=model,im_p=Blocked_image[i][j])
                                idx=idx+1
 

                print(f"       Image # {image_idx} is done!")
                bsize=np.shape(Blocked_image_feature)[0]
                

                idx=0
                df.at[image_idx,'filename']=impath
                for i in range(bsize):
                        for j in range(bsize):
                                df.at[image_idx,str(idx)]=Blocked_image_feature[i][j]
                                idx=idx+1
                image_idx=image_idx+1

        print ("No errors so far! You are an awesome programmer.")
        weight_shape=np.shape(Blocked_image_feature)[2]  
        print (f"The feature set needs {Blocked_image_feature.nbytes} bytes")
        n_zeros = np.count_nonzero(Blocked_image_feature==0)
        print (f"There are {n_zeros} zeros out of {np.shape(Blocked_image)[0]**2*156}")

        return df,weight_shape