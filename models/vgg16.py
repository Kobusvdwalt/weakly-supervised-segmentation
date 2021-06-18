import torch
import numpy as np
import cv2

from data.voc2012 import label_to_image
from models._common import ModelBase, build_vgg_features, print_params, ff, fi
from metrics.f1 import f1

class Vgg16GAP(ModelBase):
    def __init__(self, class_count=20, **kwargs):
        super(Vgg16GAP, self).__init__(**kwargs)

        self.class_count = class_count

        self.classifier = torch.nn.Sequential(
            build_vgg_features(pretrained=True, unfreeze_from=10),
            torch.nn.Conv2d(512, self.class_count, 1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Sigmoid()
        )

        self.upsample = torch.nn.Upsample(scale_factor=16, mode='nearest')
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        print_params(self.parameters(), "Classifier")

    def event(self, event):
        super().event(event)

        if event['name'] == 'minibatch':
            image = event['inputs']['image']
            label = event['labels']['classification']
            segme = event['labels']['segmentation']

            cv2.imshow('image', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
            cv2.imshow('segm', label_to_image(segme[0].clone().detach().cpu().numpy()))

            cv2.waitKey(1)

            output = self.classifier(image)
            loss = self.loss_bce(output, label)
            if self.training:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Print feedback
        if event['name'] == 'minibatch':
            self.m_f1 += f1(output.detach().cpu().numpy(), label.detach().cpu().numpy())
            self.m_loss += loss.clone().detach().cpu().item()

            m_epoch = fi(event['epoch'])
            m_batch = fi(event['batch'])
            m_phase = event['phase']
            m_f1 = ff(self.m_f1 / event['batch'])
            m_loss = ff(self.m_loss / event['batch'])
            print(f'epoch {m_epoch}, {m_phase}, batch {m_batch}, f1 {m_f1}, loss {m_loss}', end='\r')

        if event['name'] == 'phase_start':
            print('')
            self.m_f1 = 0
            self.m_loss = 0

        # Logs and save
        if event['name'] == 'phase_end':
            log_entry = {
                'f1': self.m_f1 / event['batch'],
                'loss': self.m_loss / event['batch'],
                'epoch': event['epoch'],
                'phase': event['phase']
            }
            self.logger.add(log_entry)

            if hasattr(self, 'best_f1'):
                if self.m_f1 > self.best_f1:
                    self.best_f1 = self.m_f1
                    self.save()
            else:
                self.best_f1 = self.m_f1
                self.save()

    def new_instance(self):
        return Vgg16GAP(name=self.name, class_count=self.class_count)

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