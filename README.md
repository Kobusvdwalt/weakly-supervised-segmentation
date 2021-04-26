![image not found](https://github.com/kobusvdwalt/weakly-supervised-segmentation/blob/master/_landing_page/landing.jpg?raw=true)

# Weakly Supervised Semantic Segmentation

This repo contains the code for my masters thesis titled "Segmentation From Classification".

Setup Instructions:

Step 1:
    Download the anaconda .sh file with wget 
    (wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh)

Step 2:
    Install with (bash Anaconda3-2020.07-Linux-x86_64.sh)

Step 3:
    Create new environement and install pytorch
    Select environment with source anaconda3/bin/activate and then do conda activate pytorch

Step 4:
    pip install -r requirements.txt

Step 5:
    Download and extract the datasets. There are scripts in the /datasets folder to do this

Step 6:
    Preprocess the datasets by running /data/voc2012_preprocess.py and /data/cityscapes_preprocess.py

Step 7:
    Train a classifier by running /training/[name of training script].py