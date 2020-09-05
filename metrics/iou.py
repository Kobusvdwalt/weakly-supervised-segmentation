
import cv2
import numpy as np

def iou(prediction, label):
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

    iou_result = TP / (TP + FP + FN + 1e-6)

    return iou_result

# Tests
'''
pred = np.zeros((256, 256))
label = np.zeros((256, 256))

pred[int(256*0.25): int(256*0.75)-64, int(256*0.25): int(256*0.75)] = 1
label[int(256*0.25): int(256*0.75), int(256*0.25): int(256*0.75)] = 1

cv2.imshow('pred', pred)
cv2.imshow('label', label)


m = iou(pred, label)
print(m)
cv2.waitKey(0)
'''
