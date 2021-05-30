from models._common import ModelBase
from data.voc2012 import label_to_image
import torchvision
import torch
import os, cv2
import numpy as np

from metrics.iou import iou

class DeepLab101(ModelBase):
    def __init__(self, **kwargs):
        super(DeepLab101, self).__init__(**kwargs)

        self.deeplab101 = torchvision.models.segmentation.deeplabv3_resnet101 (pretrained=True, progress=True)
        self.sigmoid = torch.nn.Sigmoid()

        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.metrics_schema = {
            'segmentation': {
                'miou': iou,
            }
        }
        
    def forward(self, inputs):
        x = inputs['image']
        x = self.deeplab101(x)['out']
        x = self.sigmoid(x)

        image = inputs['image']
        image = image[0]
        image = image.clone().detach().cpu().numpy()
        image = np.moveaxis(image, 0, 2)

        output = x[0]
        output = output.clone().detach().cpu().numpy()
        output = label_to_image(output)

        cv2.imshow('output', output)
        cv2.imshow('input', image)
        cv2.waitKey(1)

        outputs = {
            'segmentation': x
        }

        return outputs

    def backward(self, outputs, labels):
        if self.training:
            loss = self.loss_function(outputs['segmentation'], labels['segmentation'])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def should_save(self, metrics_best, metrics_last):
        metric_best = metrics_best['segmentation']['miou']
        metric_last = metrics_last['segmentation']['miou']
        return metric_last >= metric_best

    def segment(self, images, class_labels):
        x = images
        x = self.deeplab101(x)['out']
        # x = self.sigmoid(x)

        # Build label
        result = np.zeros((images.shape[0], images.shape[2], images.shape[3], 3))

        for batch_index in range(0, images.shape[0]):
            output = x.clone().detach().cpu().numpy()
            result[batch_index] = label_to_image(output[batch_index])

        return result