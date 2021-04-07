
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

def class_iou(prediction, label, axis=1):
    prediction = np.moveaxis(prediction, axis, 0)
    label = np.moveaxis(label, axis, 0)

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

    TP = np.sum(TP_im, tuple(np.arange(1, TP_im.ndim)))
    FP = np.sum(FP_im, tuple(np.arange(1, TP_im.ndim)))
    FN = np.sum(FN_im, tuple(np.arange(1, TP_im.ndim)))

    iou_result = TP / (TP + FP + FN + 1e-6)

    return iou_result