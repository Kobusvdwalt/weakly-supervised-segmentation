import sys, os, json
sys.path.insert(0, os.path.abspath('../'))

from metrics.iou import iou
from data.voc2012 import class_list, image_to_label, label_to_classes

import cv2
import numpy as np

def measure(target):
    final = np.zeros((21))
    finalCount = np.zeros((21))

    for filename in os.listdir(target):
        if (filename.split('.')[1] != 'png'):
            continue
        labelImage = cv2.imread('../datasets/voc2012/SegmentationClass/' + filename)
        outputImage = cv2.imread(target + '/' + filename)
        
        label = image_to_label(labelImage)
        output = image_to_label(outputImage)
        
        finalCount += label_to_classes(label)

        for i in range(0, 21):
            final[i] += iou(output[i], label[i])

        print(filename, end='\r')

    final /= finalCount

    measurements = {
        'miou': list(final),
        'miou_total': np.sum(final[1:]) / 20
    }

    with open(target + '/measurements.txt', 'w') as outfile:
        json.dump(measurements, outfile)

folder_name = 'output/unet_adverserial'
measure(folder_name)