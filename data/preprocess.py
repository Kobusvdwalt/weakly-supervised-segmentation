#
# This generates a set of text files to organize the data as needed
#

import cv2
import numpy as np
from voc2012 import class_list_classification

# **********************************************************
# Classification Multiclass
def multiclass_generation(source):
    labels = {}
    textFile = open('./processed/classification_multiclass_' + source + '.txt', 'w')
    for className in class_list_classification:
        f = open('../VOC2012/ImageSets/Main/' +className+ '_' + source + '.txt', 'r')
        lines = f.readlines()

        for line in lines:
            line = line.replace('\n', '')
            parts = line.split(' ')

            imageName = parts[0]

            # Fill in empty labels
            if (source == 'test'):
                labels[imageName] = ''

            # If the current image contains the current class
            if (len(parts) > 2 and parts[2] == '1'):
                if imageName in labels:
                    labels[imageName] += '|' + className
                else:
                    labels[imageName] = className

    for key, value in labels.items():
        textFile.write(key + ' ' + value +'\n')

    textFile.close()

multiclass_generation('train')
multiclass_generation('val')
multiclass_generation('test')


# **********************************************************
# Classification Binary
def binary_generation(source):
    labels = {}
    for className in class_list_classification:
        f = open('../VOC2012/ImageSets/Main/' +className+ '_' + source + '.txt', 'r')
        lines = f.readlines()
        
        textFile = open('./processed/classification_binary_' + className + '_' + source + '.txt', 'w')
        for line in lines:
            textFile.write(line.replace('  ', ' '))
        textFile.close()

binary_generation('train')
binary_generation('val')
binary_generation('test')

# **********************************************************
# Segmentation
def segmentation_generation(source):
    f = open('../VOC2012/ImageSets/Segmentation/' + source + '.txt', 'r')
    lines = f.readlines()

    textFile = open('./processed/segmentation_' + source + '.txt', 'w')
    for line in lines:
        textFile.write(line)
    textFile.close()

segmentation_generation('train')
segmentation_generation('val')
segmentation_generation('test')