import torch, cv2
import numpy as np

from data.voc2012 import label_to_image
from models._common import ModelBase, build_vgg_features
from training._common import move_to

class Vgg16GAP(ModelBase):
    def __init__(self, class_count=20, **kwargs):
        super(Vgg16GAP, self).__init__(**kwargs)
        self.image_logged = False
        self.class_count = class_count
        self.loop_count = 0

        self.features = build_vgg_features(pretrained=True, unfreeze_from=10)
        self.conv = torch.nn.Conv2d(512, self.class_count+1, 1)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat = torch.nn.Flatten(1, 3)
        self.sigm = torch.nn.Sigmoid()

        self.upsample = torch.nn.Upsample(scale_factor=16, mode='bilinear')
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def classify(self, image):
        x = self.features(image)
        segmentation = self.conv(x)
        segmentation = self.sigm(segmentation)
        
        classification = self.pool(segmentation[:, 1:])
        classification = self.flat(classification)

        segmentation = self.upsample(segmentation)

        return {
            'segmentation': segmentation,
            'classification': classification,
        }

    def event(self, event):
        super().event(event)

        if event['name'] == 'get_cam':
            image = event['inputs']['image']
            image_cu = move_to(image, self.device)

            result_cu = self.classify(image_cu)

            return result_cu['segmentation'].detach().cpu().numpy()

        if event['name'] == 'minibatch':
            image = event['inputs']['image']
            image_cu = move_to(image, self.device)

            label_classification = event['labels']['classification']
            label_classification_cu = move_to(label_classification, self.device)

            result_cu = self.classify(image_cu)
            loss = self.loss_bce(result_cu['classification'], label_classification_cu)

            if self.training:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            image_np = image.numpy()
            image_np = np.moveaxis(image_np, 1, -1)
            
            cv2.imshow('image', image_np[0])
            cv2.waitKey(1)

        if event['name'] == 'epoch_end':
            self.save()
