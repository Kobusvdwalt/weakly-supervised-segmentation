

import numpy as np
import torchvision, torch, cv2
from data.voc2012 import label_to_image
from metrics.iou import iou
from models._common import ModelBase
from models._common import fi, ff
from models._common import print_params

class DeepLab(ModelBase):
    def __init__(self, class_count=21, **kwargs):
        super(DeepLab, self).__init__(**kwargs)

        self.metric_iou = 0
        self.metric_loss = 0

        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        self.loss_cce = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        print_params(self.parameters(), "Deeplab")

    def forward(self, image):
        result = self.deeplab(image)
        return {
            'segmentation': result['out']
        }

    def event(self, event):
        if event['name'] == 'minibatch' and event['phase'] == 'train':
            image = event['inputs']['image'].to(self.device)
            label = event['labels']['segmentation_cat'].to(self.device, non_blocking=True)
            segmentation_result = self.forward(image)

            loss_cce = self.loss_cce(segmentation_result['segmentation'], label)
            loss_cce.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.metric_loss += loss_cce.item()

            print(f'batch {fi(event["batch"])}', end='\r')

        if event['name'] == 'minibatch' and event['phase'] == 'val':
            with torch.no_grad():
                image = event['inputs']['image'].to(self.device)
                label = event['labels']['segmentation_cat'].to(self.device, non_blocking=True)
                segmentation_result = self.forward(image)

                loss_cce = self.loss_cce(segmentation_result['segmentation'], label)

                segmentation = torch.softmax(segmentation_result['segmentation'], dim=1).detach().cpu().numpy()
                self.metric_iou += iou(segmentation[:, 1:], event['labels']['segmentation'][:, 1:])
                self.metric_loss += loss_cce.item()

            cv2.imshow('image', np.moveaxis(event['inputs']['image'][0].numpy(), 0, -1))
            cv2.imshow('label', label_to_image(event['labels']['segmentation'][0]))
            cv2.imshow('output', label_to_image(segmentation[0]))
            cv2.waitKey(1)
            
            print(f'batch {fi(event["batch"])}', end='\r')

        if event['name'] == 'phase_end':
            batch_count = event['batch']
            print(f' epoch {fi(event["epoch"])} phase {event["phase"]} batch {fi(event["batch"])} miou {ff(self.metric_iou/batch_count)} loss {ff(self.metric_loss/batch_count)}')
            self.metric_iou = 0
            self.metric_loss = 0
