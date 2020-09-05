import sys, os
sys.path.insert(0, os.path.abspath('../'))

from metrics.iou import iou
from data.voc2012 import class_list, LabelToImage, ImageToLabel, LabelToClasses, ClassesToWords, ThresholdClasses, AddBackgroundClass

import cv2
import numpy as np

final = np.zeros((21))
finalCount = np.zeros((21))

for filename in os.listdir('output_multiclass'):
    if (filename.split('.')[1] != 'png'):
        continue
    labelImage = cv2.imread('../VOC2012/SegmentationClass/' + filename)
    outputImage = cv2.imread('output_multiclass/' + filename)
    
    label = ImageToLabel(labelImage)
    output = ImageToLabel(outputImage)
    
    finalCount += LabelToClasses(label)

    for i in range(0, 21):
        final[i] += iou(output[i], label[i])

    print(filename)

final /= finalCount
print(final)
print(np.sum(final[1:]) / 20)