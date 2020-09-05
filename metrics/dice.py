

import cv2
import numpy as np

def dice(prediction, label):
    # Threshold to get either 1 or 0
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0

    label[label >= 0.5] = 1
    label[label < 0.5] = 0

    # Compute True Positives, False Positives and False Negatives
    TP_im = prediction * label
    FP_im = prediction - label
    FN_im = label - prediction

    FP_im[FP_im < 0] = 0
    FN_im[FN_im < 0] = 0

    TP = np.sum(TP_im)
    FP = np.sum(FP_im)
    FN = np.sum(FN_im)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1 = (2 * precision * recall) / (precision + recall)

    return F1

# Tests
pred = np.zeros((256, 256))
label = np.zeros((256, 256))

pred[int(256*0.25): int(256*0.75), int(256*0.25): int(256*0.75)] = 1
label[int(256*0.25): int(256*0.75), int(256*0.25): int(256*0.75)] = 1

cv2.imshow('pred', pred)
cv2.imshow('label', label)


m = dice(pred, label)
print(m)
cv2.waitKey(0)