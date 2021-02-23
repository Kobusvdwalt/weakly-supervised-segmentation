import os, json

from metrics.iou import iou
from data.voc2012 import class_list, image_to_label, label_to_classes

from artifacts import artifact_manager

import cv2
import numpy as np

def measure_model(target, output_file):
    final = np.zeros((21))
    finalCount = np.zeros((21))
    count = 0
    for filename in os.listdir(target):
        if (filename.split('.')[1] != 'png'):
            continue
        labelImage = cv2.imread('datasets/voc2012/SegmentationClass/' + filename)
        outputImage = cv2.imread(target + '/' + filename)
        
        label = image_to_label(labelImage)
        output = image_to_label(outputImage)
        
        finalCount += label_to_classes(label)

        for i in range(0, 21):
            final[i] += iou(output[i], label[i])

        print("Measure No: " + str(count), end='\r')
        count += 1
    print()
    final /= finalCount + 0.0000001

    measurements = {
        'miou': list(final),
        'miou_total': np.sum(final[1:]) / 20
    }

    with open(artifact_manager.instance.getArtifactDir() + output_file + '.txt', 'w') as outfile:
        json.dump(measurements, outfile)

def measure(folder_name, output_file):
    measure_model(folder_name, output_file)