

import numpy as np
import torchvision, torch, cv2
import wandb
from data.voc2012 import label_to_image
from models._common import ModelBase
from models._common import fi, ff
from models._common import print_params
from metrics.iou import class_iou, iou
from sklearn import metrics
from training.config_manager import config_manager

class DeepLab(ModelBase):
    def __init__(self, class_count=21, **kwargs):
        super(DeepLab, self).__init__(**kwargs)

        config = config_manager.getConfig()
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=config.semseg_pretrained, progress=True)
        self.loss_cce = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)

        print_params(self.parameters(), "Deeplab")

    def forward(self, image):
        result = self.deeplab(image)
        return result['out']

    def event(self, event):
        if event['name'] == 'get_semseg':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            return self.forward(image_cu)

        if event['name'] == 'minibatch' and event['phase'] == 'train':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_cu = event['labels']['segmentation'].cuda(non_blocking=True)
            label_cu = torch.argmax(label_cu, 1).long()

            # cv2.imshow('amaxed', label_cu[0].detach().float().cpu().numpy() / 21)

            segmentation_result = self.forward(image_cu)
            loss = self.loss_cce(segmentation_result, label_cu)

            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()


            if event['batch'] % 2 == 0:
                datapacket = event['data']
                image = event['inputs']['image'].detach().numpy()
                label = event['labels']['segmentation'].detach().numpy()
                pseudo = event['labels']['pseudo'].detach().numpy()
                predi = segmentation_result.detach().cpu().numpy()
                batch_size = image.shape[0]

                log = {
                    'loss': loss.detach().cpu().numpy(),
                    'acc': 0,
                    'mapr': 0,
                    'miou_macro': 0,
                    'p_miou_macro': 0,
                }

                for i in range(0, batch_size):
                    content_width = datapacket['content_width'][i].detach().numpy()
                    content_height = datapacket['content_height'][i].detach().numpy()

                    content_image = image[i, :, 0:content_height, 0:content_width]
                    content_image_vis = np.moveaxis(content_image, 0, -1)
                    
                    content_label = label[i, :, 0:content_height, 0:content_width]
                    content_predi = predi[i, :, 0:content_height, 0:content_width]

                    content_pseudo = pseudo[i, :, 0:content_height, 0:content_width]

                    cv2.imshow('content_pseudo', label_to_image(content_pseudo))
                    cv2.imshow('content_label', label_to_image(content_label))
                    cv2.imshow('content_predi', label_to_image(content_predi))
                    cv2.imshow('content_image', content_image_vis)
                    cv2.waitKey(1)

                    log['acc'] += metrics.accuracy_score(np.argmax(content_label, 0).flatten(), np.argmax(content_predi, 0).flatten())
                    log['mapr'] += metrics.average_precision_score(content_label[1:].flatten(), content_predi[1:].flatten())
                    log['miou_macro'] += metrics.jaccard_score(np.argmax(content_label, 0).flatten(), np.argmax(content_predi, 0).flatten(), average='macro')
                    log['p_miou_macro'] += metrics.jaccard_score(np.argmax(content_label, 0).flatten(), np.argmax(content_pseudo, 0).flatten(), average='macro')

                log['acc'] /= batch_size
                log['mapr'] /= batch_size
                log['miou_macro'] /= batch_size
                log['p_miou_macro'] /= batch_size

                wandb.log(log)

        if event['name'] == 'epoch_end':
            print('')
            self.save()