import torch
import numpy as np
import cv2

from data.voc2012 import label_to_image
from models._common import ModelBase, build_vgg_features, print_params, ff, fi
from metrics.f1 import f1

class Blobber(torch.nn.Module):
    def __init__(self, kernel_size = 3, iterations = 1):
        super().__init__()
        self.iterations = iterations
        self.blob_conv = torch.nn.Conv2d(1, 1, kernel_size, padding=int((kernel_size-1)/2))
        self.blob_sigm = torch.nn.Sigmoid()

        self.blob_conv.bias.requires_grad = False
        self.blob_conv.weight.requires_grad = False
        self.blob_conv.bias.data.fill_(0)
        self.blob_conv.weight.data.fill_(1.0/(self.blob_conv.weight.shape[2]**2))

    def forward(self, inputs):
        for i in range (0, self.iterations):
            # Expand
            blob = self.blob_conv(inputs)
            blob = self.blob_sigm((blob - 0.01) * 1000)
            # Shrink
            blob = self.blob_conv(blob)
            blob = self.blob_sigm((blob - 0.9) * 1000)
        return blob

class Vgg16GAP(ModelBase):
    def __init__(self, **kwargs):
        super(Vgg16GAP, self).__init__(**kwargs)

        self.classifier = torch.nn.Sequential(
            build_vgg_features(),
            torch.nn.Conv2d(512, 20, 1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Sigmoid()
        )

        self.blob = Blobber(kwargs['blob_size'])

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

            mask, _ = torch.max(segme[:, 1:], dim=1, keepdim=True)
            mask = self.blob(mask)
            image = image * (1 - mask)

            cv2.imshow('segme', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
            cv2.imshow('image', label_to_image(segme[0].clone().detach().cpu().numpy()))
            cv2.imshow('maskz', mask[0, 0].clone().detach().cpu().numpy())

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
                'f1': self.m_f1,
                'loss': self.m_loss,
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