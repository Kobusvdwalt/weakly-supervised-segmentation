
import cv2
import numpy as np
from data.voc2012 import image_to_label, label_to_classes, get_augmentation, label_smoothing

def generate_split(source):
    f = open('datasets/voc2012/ImageSets/Segmentation/' + source + '.txt', 'r')
    lines = f.readlines()

    # Return the data as two arrays for easy access
    images_array = []
    labels_array = []
    for image_name in lines:
        image_name = image_name.replace('\n', '')
        label = cv2.imread('datasets/voc2012/SegmentationClass/' + image_name + '.png')
        label = image_to_label(label)

        classification = np.delete(label_to_classes(label), 0)
        class_count = np.sum(classification)

        for class_index, class_value in enumerate(classification):
            label_index = class_index + 1
            if class_value == 1:
                cv2.imwrite('datasets/voc2012/Masks/' + image_name + '_' + str(label_index) + '.png', label[label_index] * 255)

        images_array.append(image_name)
        labels_array.append(image_name)

def generate():
    generate_split('train')