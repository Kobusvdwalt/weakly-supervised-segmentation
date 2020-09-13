import numpy as np

def f1(prediction, label):
    # Threshold to get either 1 or 0
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0

    label[label >= 0.5] = 1
    label[label < 0.5] = 0

    # Compute True Positives, False Positives, True Negatives and False Negatives
    TP_im = prediction * label
    FP_im = prediction - label
    TN_im = (1 - label) * (1 - prediction)
    FN_im = label - prediction

    FP_im[FP_im < 0] = 0
    FN_im[FN_im < 0] = 0

    TP = np.sum(TP_im)
    FP = np.sum(FP_im)
    TN = np.sum(TN_im)
    FN = np.sum(FN_im)

    precision = TP/(TP+FP+ 1e-6)
    recall = TP/(TP+FN+ 1e-6)

    return 2*(recall * precision) / (recall + precision + 1e-6)