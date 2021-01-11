from data.voc2012 import label_to_image
import cv2
import torchvision
import torch
import numpy as np
from models.model_base import ModelBase

from metrics.f1 import f1

class Vgg16GAP(ModelBase):
    def __init__(self, **kwargs):
        super(Vgg16GAP, self).__init__(**kwargs)

        self.vgg = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg.features = self.vgg.features[:-1]
        self.vgg.avgpool = None
        self.vgg.classifier = None
        self.upsample = torch.nn.Upsample(scale_factor=16, mode='nearest') # , align_corners=True

        # Unfreeze last conv layer
        total = 0
        count = 0
        unfreeze = 2
        for param in self.vgg.parameters():
            total += 1
        for param in self.vgg.parameters():
            if (count >= total-unfreeze*2):
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1

        self.conv = torch.nn.Conv2d(512, 20, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, inputs):
        x = inputs['image']
        x = self.vgg.features(x)
        x = self.conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.sigmoid(x)

        outputs = {
            'classification': x
        }

        return outputs

    def backward(self, outputs, labels):
        if self.training:
            loss = self.loss_function(outputs['classification'], labels['classification'])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def metrics(self, outputs, labels):
        metrics = {
            'classification': {
                'f1': f1,
            }
        }
        metrics_output = {}
        for output_key in metrics:
            metrics_output[output_key] = {}
            for metric_name in metrics[output_key]:
                metric_func = metrics[output_key][metric_name]
                metric_result = metric_func(outputs[output_key].cpu().detach().numpy(), labels[output_key].cpu().detach().numpy())
                metrics_output[output_key][metric_name] = metric_result

        return metrics_output

    def should_save(self, metrics_best, metrics_last):
        metric_best = metrics_best['classification']['f1']
        metric_last = metrics_last['classification']['f1']
        return metric_best >= metric_last

    def segment(self, images, class_labels):
        # Extract last layer features
        x = images
        x = self.vgg.features(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = self.upsample(x)

        # Build label
        result = np.zeros((images.shape[0], images.shape[2], images.shape[3], 3))

        for batch_index in range(0, images.shape[0]):
            features = x[batch_index].clone().detach().cpu().numpy()
            label = np.zeros((features.shape[0]+1, features.shape[1], features.shape[2]))

            # Copy masks based on classification label
            for feature_index in range(0, features.shape[0]):
                if class_labels[batch_index, feature_index] > 0.5:
                    label[feature_index+1] = features[feature_index]


            # Compute background mask
            summed = np.mean(features, 0)
            summed[summed > 1] = 1
            summed[summed < 0] = 0
            label[0] = (1 - summed) * 0.5

            result[batch_index] = label_to_image(label)

        return result