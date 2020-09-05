from torch.utils.data import DataLoader
from data import PascalVOCSegmentation
from deeplab import model
from pascal_helper import LabelToImage

import numpy as np
import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pascal_val = PascalVOCSegmentation(source='val')
data = DataLoader(pascal_val, batch_size=4, shuffle=False, num_workers=0)
model.eval()

for images, labels in data:
    inputs = images.permute(0, 3, 1, 2)

    inputs = inputs.float()
    labels = labels.float()

    # Predict
    outputs = model(inputs)['out']

    # Convert to numpy
    images_np = images.numpy()
    labels_np = labels.numpy()
    outputs_np = outputs.data.numpy()

    for sample in range(0, outputs_np.shape[0]):
        classPrediction = np.argmax(outputs_np[sample])
        labelPrediction = np.argmax(labels_np[sample])

        cv2.imshow('image', images_np[sample])
        cv2.imshow('label', LabelToImage(labels_np[sample]))
        cv2.imshow('output', LabelToImage(outputs_np[sample]))
        cv2.waitKey(0)
