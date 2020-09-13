#
# This file generates train and val text files in the data folder which is used for training
#

import cv2
import numpy as np
from data import classListArray

labels = {}

def buildText(className, source):
    labels = {}
    textFile = open('./data/' + className + '_' + source + '.txt', 'w')
    f = open('../voc2012/ImageSets/Main/' +className+ '_' + source + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        parts = line.split(' ')

        imageName = parts[0]

        if (len(parts) > 2):
            if (parts[2] == '1'):
                labels[imageName] = className

    for key, value in labels.items():
        textFile.write(key +'\n')
    textFile.close()

for className in classListArray:
    buildText(className, 'train')
    buildText(className, 'val')