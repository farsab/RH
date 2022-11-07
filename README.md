# RH

> This is the implementation of my paper.

## Table of Contents
* [General Information](#general-information)
* [Requirements](#requirements)
* [Acknowledgements](#acknowledgements)


## General Information
Implementation of the paper
The repository contains core functions for feature genration, weight calculation and binary search tree construction.
To use your own dataset, you need to place the images in the "dataset" folder and also replace the actual path in the main file.
Feel free to contact me or create a issue if you find bugs or have any questions.
The datasets are standard benchmarks without any changes. 
To run the code, you need three datasets: Cifar 10, NUS-WIDE and ImageNet. All these datasets can be downloaded from their official websites.
- Cifar10: https://www.cs.toronto.edu/~kriz/cifar.html
- NUS-WIDE: https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
- ImageNet: https://www.image-net.org/download.php

Once the features for training, test and query(ies) are calculated, you would opt in to save them in addition to save weight matrix.

If you want to have the features and weight matrix saved, disable "LOAD" variable by LOAD=False
Once you have saved them, you should set "LOAD" by LOAD=True. In this case, you do not have to re-run the feature extraction and weight calculation.
However, keep in mind that the the results you would not get exactly the same as resutls shown in the paper because of nature of randomness as during the training the training set is shuffled. This different is negligble.
## Requirements
- Python 3.8
- Keras 2.9.0
- Pandas
- Numpy

List of the required packages are in requirements.txt.
To use it:
- Create a virtual enviroment
- Activate it
- Run: pip3 install -r requirements.txt

## Acknowledgements
I used code from this paper:
Guang-Hai Liu, Jing-Yu Yang,
Content-based image retrieval using color difference histogram,
Pattern Recognition,
Volume 46, Issue 1,
https://doi.org/10.1016/j.patcog.2012.06.001.
- The color feature was based on https://github.com/AdityaShaha/CBIR-Using-CDH. I did not change the code. The only change that is made is to normalize the feature vector to avoid to one of the feature groups diminish the influence of the other group. Without the normalization, the feature set acquired from Gabor would be less effective due to be smaller values than that of color features.

For comparison purposes, codes available in below respositories are used:
- https://github.com/swuxyj/DeepHash-pytorch
- https://github.com/kevinlin311tw/cvpr16-deepbit
- https://github.com/ssppp/GreedyHash

