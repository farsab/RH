import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from . import Block_feat_dataframe


def HNN_blocks_weights_calculation_LLF(im_p,im_size=256,block_num=4,test_im=0,show_test=0,show_weights=False,save=''):
        g1err_total=g2err_total=NaNerr_total=0

        df,weight_shape,g1err,g2err,NaNerr,df_err=Block_feat_dataframe.Block_feat_dataframe_LLF(im_p,im_size=im_size,block_num=block_num,test_im=test_im,show_test=show_test)
        g1err_total=g1err_total+g1err
        g2err_total=g2err_total+g2err
        NaNerr_total=NaNerr_total+NaNerr




        Num_images=df.shape[0] # Number of images for averaging Block_weight
        Num_feat=df.shape[1] # Number of the features
        eta = 1./Num_images
        Block_weight = np.zeros((Num_feat-1,weight_shape,weight_shape)) #-1 because we have 17 columns in df. The first column is the filename
        odx=1
        img_cnt=0
        for Nimage in range (Num_images):
                img_cnt=img_cnt+1
                for block_num in range(1,Num_feat):
                        Bfeat_org=df.iloc[Nimage,block_num]
                        Bfeat_trans=np.array(Bfeat_org.reshape(-1,1))
                        Calc=np.outer(Bfeat_org,Bfeat_trans)
                        Block_weight[block_num-1][:][:]=Block_weight[block_num-1][:][:]+Calc # -1 because the block_num starts from 1 whereas in Block_weight it should start from 0
                        del (Calc)
                        odx=odx+1
        print (f"Calculation of the weight matrix is done for {img_cnt} images. Shape of Block_weight is {np.shape(Block_weight)}")
        if show_weights:
                fig,axs=plt.subplots(4,4)
                idx=1
                for i in range(4):
                        for j in range(4):
                                axs[i,j].set_title("Block "+str(idx),fontsize=4)
                                axs[i,j].set_xlabel('Features',fontsize=4)
                                axs[i,j].set_ylabel('',fontsize=4)
                                axs[i,j].plot(Block_weight[idx-1])
                                
                                idx=idx+1

                fig.tight_layout()
                
                if save!='':
                        plt.savefig(save,dpi=300)
                plt.show()

        return Block_weight,g1err_total,g2err_total,NaNerr_total,df_err


def HNN_blocks_weights_calculation_HLF(model,im_p,im_size=256,block_num=4,test_im=0,show_test=0,show_weights=False,save=''):


        df,weight_shape=Block_feat_dataframe.Block_feat_dataframe_HLF(model=model,im_p=im_p,im_size=im_size,block_num=block_num,test_im=test_im,show_test=show_test)





        Num_images=df.shape[0] # Number of images for averaging Block_weight
        Num_feat=df.shape[1] # Number of the features
        eta = 1./Num_images
        Block_weight = np.zeros((Num_feat-1,weight_shape,weight_shape)) #-1 because we have 17 columns in df. The first column is the filename
        odx=1
        img_cnt=0
        for Nimage in range (Num_images):
                img_cnt=img_cnt+1
                for block_num in range(1,Num_feat):
                        Bfeat_org=df.iloc[Nimage,block_num]
                        Bfeat_trans=np.array(Bfeat_org.reshape(-1,1))
                        Calc=np.outer(Bfeat_org,Bfeat_trans)
                        Block_weight[block_num-1][:][:]=Block_weight[block_num-1][:][:]+Calc # -1 because the block_num starts from 1 whereas in Block_weight it should start from 0
                        del (Calc)
                        odx=odx+1
        print (f"Calculation of the weight matrix is done for {img_cnt} images. Shape of Block_weight is {np.shape(Block_weight)}")
        if show_weights:
                fig,axs=plt.subplots(4,4)
                idx=1
                for i in range(4):
                        for j in range(4):
                                axs[i,j].set_title("Block "+str(idx),fontsize=4)
                                axs[i,j].set_xlabel('Features',fontsize=4)
                                axs[i,j].set_ylabel('',fontsize=4)
                                axs[i,j].plot(Block_weight[idx-1])
                                
                                idx=idx+1

                fig.tight_layout()
                
                if save!='':
                        plt.savefig(save,dpi=300)
                plt.show()

        return Block_weight