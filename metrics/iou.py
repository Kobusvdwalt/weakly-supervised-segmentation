
import numpy as np

def iou(prediction, label):
    # Threshold to get either 1 or 0
    ti = prediction >= 0.5
    prediction = np.zeros(prediction.shape)
    prediction[ti] = 1

    ti = label >= 0.5
    label = np.zeros(label.shape)
    label[ti] = 1

    positive = 1
    negative = 0

    TP = np.sum(np.logical_and(prediction == positive, label == positive))
    # TN = np.sum(np.logical_and(prediction == negative, label == negative))
    FP = np.sum(np.logical_and(prediction == positive, label == negative))
    FN = np.sum(np.logical_and(prediction == negative, label == positive))

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