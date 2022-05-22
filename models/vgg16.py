
import torch
import numpy as np
import wandb

from models._common import ModelBase, build_vgg_features
from sklearn import metrics
from training.config_manager import config_manager

class Vgg16GAP(ModelBase):
    def __init__(self, class_count=20, **kwargs):
        super(Vgg16GAP, self).__init__(**kwargs)
        self.class_count = class_count

        config = config_manager.getConfig()
        self.features = build_vgg_features(pretrained=config.classifier_pretrained, unfreeze_from=config.classifier_pretrained_unfreeze)
        self.drop_3 = torch.nn.Dropout2d(p=0.5)
        self.conv_3 = torch.nn.Conv2d(512, self.class_count, 1, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat = torch.nn.Flatten(1, 3)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

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

        if event['name'] == 'minibatch':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_classification_cu = event['labels']['classification'].cuda(non_blocking=True)

            result_cu = self.classify(image_cu)
            loss = self.loss(result_cu, label_classification_cu)

            if self.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                label = event['labels']['classification'].detach().numpy().flatten()
                label[label > 0.5] = 1
                label[label <= 0.5] = 0
                pred = torch.sigmoid(result_cu).detach().cpu().numpy().flatten()
                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0

                m_loss = loss.detach().cpu().numpy()
                m_accuracy = metrics.accuracy_score(label, pred)
                m_f1 = metrics.f1_score(label, pred, average='macro')
                m_precision = metrics.precision_score(label, pred, average='macro')
                m_recall = metrics.recall_score(label, pred, average='macro')

                wandb.log({
                    'loss': m_loss,
                    'accuracy': m_accuracy,
                    'f1': m_f1,
                    'precision': m_precision,
                    'recall': m_recall,
                })

        if event['name'] == 'epoch_end':
            self.save()
