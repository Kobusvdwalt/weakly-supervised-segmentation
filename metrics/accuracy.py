import numpy as np

def accuracy(prediction, label):
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

    return (TP+TN)/(TP+FP+FN+TN+1e-6)
