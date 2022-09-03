import numpy as np
from .Color_CDH import color_util
from .Gabor import Gabor_util


def compute_all_feautres(im_p):
    NaNerr=0
    hist,g1err,g2err=color_util.compute_color_feature(im_p,lnum=10,anum=3,bnum=3,onum=18,test=False)
    gabor_feature=Gabor_util.Gabor_features(im_p,test=0,rotat=90)
    feat=np.concatenate((hist,gabor_feature),axis=0)
    if len(np.argwhere(np.isnan(feat)))>0:
        NaNerr=1

    np.where(np.isnan(feat), 0.0, feat)
    if max(feat)==0:
        feat=feat/1.0
    else:
        feat=feat/max(abs(feat))


    return feat,g1err,g2err,NaNerr