import sys, os
sys.path.insert(0, os.path.abspath('../'))

from torch.utils.data import DataLoader
from classification.data import PascalVOCClassification
from classification.data import classListArray
from classification.models.vgg_gap import vgg

import numpy as np
import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pascal_val = PascalVOCClassification(source='val')
data = DataLoader(pascal_val, batch_size=16, shuffle=True, num_workers=0)

vgg.train()

for images, labels in data:
    inputs = images.permute(0, 3, 1, 2)

    inputs = inputs.float()
    labels = labels.float()

    # Get final conv layer activation
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    vgg.vgg16.features[28].register_forward_hook(get_activation('last_conv'))

    # Predict
    outputs = vgg(inputs)

    # Convert to numpy
    images_np = images.numpy()
    labels_np = labels.numpy()
    outputs_np = outputs.data.numpy()
    weights_np = vgg.dense1.weight.data.numpy()
    activations_np = activation['last_conv'].numpy()

    for sample in range(0, activations_np.shape[0]):
        classPrediction = np.argmax(outputs_np[sample])
        labelPrediction = np.argmax(labels_np[sample])
        print('predict: ' + classListArray[classPrediction])
        print('label: ' + classListArray[labelPrediction])
        cam = np.zeros(shape=(activations_np.shape[2], activations_np.shape[3]))
        for w in range(0, len(weights_np[classPrediction])):
            cam += activations_np[sample, w] * weights_np[classPrediction, w]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 0.000000001)
        out = cv2.resize(cam, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('input', images_np[sample])
        cv2.imshow('cam', out) 
        cv2.waitKey(0)

    exit()