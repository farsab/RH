import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# The code is adopted from https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
def maxpool1d(x,stride):
    
    maxp=[]
    for i in range(0,np.shape(x)[0],stride):

        maxp.append(np.max(x[i:i+stride]))
        if (i+stride+1==np.shape(x)[0]-1):
            break

    return maxp

def compute_all_feautres_HLF(model,im_p):
    image = smart_resize(im_p,(224,224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    feat_HL = model.predict(image)
    feat_HL = np.squeeze(feat_HL)
    feat_HL = maxpool1d(feat_HL,stride=26)
    feat_HL = feat_HL[:156]
    
    return feat_HL