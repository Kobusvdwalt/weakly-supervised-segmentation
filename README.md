![image not found](https://github.com/kobusvdwalt/weakly-supervised-segmentation/blob/master/_landing_page/landing.jpg?raw=true)

# Weakly Supervised Semantic Segmentation

This repo contains the code for my masters thesis titled "Segmentation From Classification".

Dependancies:
* OpenCV
* Pytorch


Setup Instructions:

Step 1:
    Download the anaconda .sh file with wget 
    (wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh)

Step 2:
    Install with (bash Anaconda3-2020.07-Linux-x86_64.sh)

Step 3:
    Create new environement and install pytorch

Step 4:
    Install albumentations

Step 5:
    Download and extract the datasets. There are scripts in the /datasets folder to do this

Step 6:
    Preprocess the datasets by running /data/voc2012_preprocess.py and /data/cityscapes_preprocess.py

Step 7:
    Train a classifier by running /classification_multiclass/train.py