from torch.utils.data import DataLoader
from data import PascalVOCClassification
from data import classListArray
from vgg_cam import vgg

import numpy as np
import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pascal_val = PascalVOCClassification(source='val')
data = DataLoader(pascal_val, batch_size=16, shuffle=True, num_workers=0)

vgg.train()

for images, labels in data:
    inputs = images.permute(0, 3, 1, 2)

    inputs_var = torch.autograd.Variable(inputs.float())
    inputs_var.requires_grad = True
    labels = labels.float()

    # Predict
    outputs = vgg(inputs_var)
    outputs.backward(labels)
    grads = inputs_var.grad

    # Convert to numpy
    outputs_np = outputs.data.numpy()
    images_np = images.numpy()
    labels_np = labels.numpy()
    grads_np = grads.numpy()

    for sample in range(0, outputs_np.shape[0]):
        classPrediction = np.argmax(outputs_np[sample])
        labelPrediction = np.argmax(labels_np[sample])
        print('predict: ' + classListArray[classPrediction])
        print('label: ' + classListArray[labelPrediction])
        print('label: ' + str(labels_np[sample]))
        grad = grads_np[sample]
        grad = np.moveaxis(grad, 0, 2)
        print(grad.shape)

        threshold_indices = grad < 0
        grad[threshold_indices] = 0
        arr_min, arr_max = np.min(grad), np.max(grad)
        grad = (grad - arr_min) / (arr_max - arr_min + 0.000000001)
        # grad = grad / (arr_max + 0.000000001)
        cv2.imshow('input', images_np[sample])
        cv2.imshow('cam', grad) 
        cv2.waitKey(0)
    exit()