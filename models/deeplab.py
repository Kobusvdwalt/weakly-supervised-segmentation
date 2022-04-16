

import numpy as np
import torchvision, torch, cv2
from data.voc2012 import label_to_image
from models._common import ModelBase
from models._common import fi, ff
from models._common import print_params

class DeepLab(ModelBase):
    def __init__(self, class_count=21, **kwargs):
        super(DeepLab, self).__init__(**kwargs)

        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
        self.loss_cce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        print_params(self.parameters(), "Deeplab")

    def forward(self, image):
        result = self.deeplab(image)
        return {
            'segmentation': result['out']
        }

    def event(self, event):
        if event['name'] == 'get_semseg':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            segmentation_result = self.forward(image_cu)
            return segmentation_result['segmentation']

        if event['name'] == 'minibatch' and event['phase'] == 'train':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_cu = event['labels']['segmentation'].cuda(non_blocking=True)
            segmentation_result = self.forward(image_cu)

            loss = self.loss_cce(torch.sigmoid(segmentation_result['segmentation']), label_cu)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            image = event['inputs']['image'].detach().numpy()
            image = image[0]
            image = np.moveaxis(image, 0, -1)
            cv2.imshow('image', image)

            label = event['labels']['segmentation'].detach().numpy()
            label = label[0]
            label = label_to_image(label)
            cv2.imshow('label', label)

            prediction = segmentation_result['segmentation']
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction[0]
            prediction = label_to_image(prediction)
            cv2.imshow('prediction', prediction)

            cv2.waitKey(1)

            print(event['batch'])
        
        if event['name'] == 'epoch_end':
            print('')
            self.save()