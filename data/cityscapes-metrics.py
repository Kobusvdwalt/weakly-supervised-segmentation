
# TODO: NOW WE CAN MEASURE ALL OF THE DETAILS OF THE CITYSCAPES DATASET
import sys, os
import numpy as np
import cv2
sys.path.insert(0, os.path.abspath('../'))

from data.cityscapes import get_class_count, words_to_classes


split = 'train'

file = open('./output/cityscapes_classification_' + split + '.txt', 'r')
lines = file.readlines()

class_count_store = np.zeros(shape=(get_class_count()))
for line in lines:
    image_path = line.split(' ')[0].replace('\n', '')
    label_words = line.split(' ')[1].replace('\n', '')
    classes = words_to_classes(label_words)
    class_count_store += classes

    # Which images don't have "ROAD"
    # if (classes[0] == 0):
    #     label_path = image_path.replace('_leftImg8bit', '_gtFine_color')
    #     label_path = label_path.replace('leftImg8bit', 'gtFine')

    #     image = cv2.imread('../datasets/cityscapes/' + image_path)
    #     label = cv2.imread('../datasets/cityscapes/' + label_path)

print('Image Count:')
print(len(lines))

print('Class Count:')
print(class_count_store)