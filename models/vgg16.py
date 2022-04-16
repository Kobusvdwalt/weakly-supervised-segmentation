
import torch
import numpy as np

from data.voc2012 import label_to_image
from models._common import ModelBase, build_vgg_features
from training._common import move_to
import torch.nn.functional as F

class Vgg16GAP(ModelBase):
    def __init__(self, class_count=20, **kwargs):
        super(Vgg16GAP, self).__init__(**kwargs)
        self.class_count = class_count

        self.features = build_vgg_features(pretrained=True, unfreeze_from=10)
        self.drop_3 = torch.nn.Dropout2d(p=0.5)
        self.conv_3 = torch.nn.Conv2d(512, self.class_count, 1, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat = torch.nn.Flatten(1, 3)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.upsample = torch.nn.Upsample(scale_factor=16, mode='bilinear')
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.timer = None

    def segment(self, image):
        x = self.features(image)
        x = self.conv_3(x)
        x = torch.relu(x)
        x = torch.sqrt(x)
        return x

    def classify(self, image):
        x = self.features(image)
        x = self.conv_3(self.drop_3(x))
        x = self.pool(x)
        x = self.flat(x)
        return x

    def event(self, event):
        super().event(event)

        if event['name'] == 'get_cam':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            result_cu = self.segment(image_cu)
            return result_cu.detach().cpu().numpy()

        if event['name'] == 'phase_start':
            self.metrics = {
                'loss': 0,
                'f1': 0,
                'acc': 0,
            }

        if event['name'] == 'minibatch':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_classification_cu = event['labels']['classification'].cuda(non_blocking=True)

            result_cu = self.classify(image_cu)
            loss = self.loss(result_cu, label_classification_cu)

            if self.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Print metrics
            if event['batch'] % 3 == 0:
                self.metrics['loss'] += loss.detach().cpu().numpy()

                output = {
                    'epoch': "{0:05d}".format(event['epoch']),
                    'batch': "{0:05d}".format(event['batch']),
                    'loss': "{:10.4f}".format(self.metrics['loss'] / (event['batch'] /3))
                }

                metricsLog = ''
                for key in output:
                    metricsLog += key + ' : ' + output[key] + ' '
                print(metricsLog, end='\r')

        if event['name'] == 'epoch_end':
            print('')
            self.save()
