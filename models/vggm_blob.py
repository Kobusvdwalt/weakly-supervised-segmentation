import torch
import numpy as np
import cv2

from data.voc2012 import label_to_image
from models._common import ModelBase, build_vgg_features, print_params, ff, fi
from metrics.f1 import f1
from models.blobber import Blobber

class VggMBlob(ModelBase):
    def __init__(self, **kwargs):
        super(VggMBlob, self).__init__(**kwargs)

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, 1),
            build_vgg_features(),
            torch.nn.Conv2d(512, 20, 1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Sigmoid()
        )

        # relu_angle = 0.05
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 16, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.Conv2d(16, 16, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.MaxPool2d(2),

        #     torch.nn.Conv2d(16, 32, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.Conv2d(32, 32, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.MaxPool2d(2),

        #     torch.nn.Conv2d(32, 64, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.Conv2d(64, 64, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.MaxPool2d(2),

        #     torch.nn.Conv2d(64, 128, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.Conv2d(128, 128, 3, padding=1),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #     torch.nn.Flatten(1, 3),

        #     torch.nn.Linear(128, 64),
        #     torch.nn.LeakyReLU(relu_angle),
        #     torch.nn.Linear(64, 20),
        #     torch.nn.Sigmoid()
        # )

        self.blob_size = kwargs['blob_size']
        if self.blob_size > 0:
            self.blob = Blobber(self.blob_size)

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
            if self.blob_size > 0:
                mask = self.blob(mask)


            cv2.imshow('maskz', mask[0, 0].clone().detach().cpu().numpy())
            cv2.imshow('segme', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
            cv2.imshow('image', label_to_image(segme[0].clone().detach().cpu().numpy()))

            cv2.waitKey(1)
            output = self.classifier(mask)
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