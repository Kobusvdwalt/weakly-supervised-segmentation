
import cv2
import numpy as np
import torch
import wandb
from data.voc2012 import classes_to_words

from models._common import ModelBase, build_vgg_features, print_params
from training.config_manager import config_manager
from sklearn import metrics
from metrics.f1 import f1

class EraseVGG16(ModelBase):
    def __init__(self, class_count=20, **kwargs):
        super(EraseVGG16, self).__init__(**kwargs)
        self.class_count = class_count

        self.features = build_vgg_features(pretrained=True, unfreeze_from=10)
        self.conv = torch.nn.Conv2d(512, self.class_count, 1)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.leakyrelu = torch.nn.LeakyReLU()

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        print_params(self.parameters(), 'class')

        self.train()

    def classify(self, image):
        x = self.features(image)
        x = self.conv(x)
        x = self.leakyrelu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1, 3)
        return x

    def event(self, event):
        super().event(event)

        if event['name'] == 'minibatch':
            image_cu = event['inputs']['image'].float().cuda(non_blocking=True)
            label_classification_cu = event['labels']['classification'].float().cuda(non_blocking=True)

            result = self.classify(image_cu)

            cv2.imshow('image', np.moveaxis(image_cu[0].detach().cpu().numpy(), 0, -1))
            cv2.waitKey(1)

            if self.training:
                self.optimizer.zero_grad()
                loss = self.loss(result, label_classification_cu)
                loss.backward()
                self.optimizer.step()

                label = event['labels']['classification'].clone().detach().cpu().numpy()
                label[label > 0.5] = 1
                label[label <= 0.5] = 0

                pred = torch.sigmoid(result.clone()).detach().cpu().numpy()
                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0

                label_flat = label.flatten()
                pred_flat = pred.flatten()
                
                m_loss = loss.clone().detach().cpu().numpy()
                m_recall = metrics.recall_score(label_flat, pred_flat, average='binary', zero_division=0)
                m_precision = metrics.precision_score(label_flat, pred_flat, average='binary', zero_division=0)
                m_f1 = metrics.f1_score(label_flat, pred_flat, average='binary', zero_division=0)

                print('m_recall: ', m_recall)
                print('m_precision: ', m_precision)
                print('m_f1: ', m_f1) 

                # print('m_precision: ', m_precision)

                # print(classes_to_words(np.append(np.array([0]), pred[0])))
                # print(classes_to_words(np.append(np.array([0]), label[0])))
                # print('-----------------')

                wandb.log({
                    'loss': m_loss,
                    'f1': m_f1,
                    'precision': m_precision,
                    'recall': m_recall,
                })


                # result_np = torch.sigmoid(result.clone().detach().cpu()).numpy()
                # print(result_np[0])

                
                # print(classes_to_words(np.append(np.array([0]), result_np[0])))
                # print(classes_to_words(np.append(np.array([0]), label_classification_cu[0].clone().detach().cpu().numpy())))
                # 
                



            #     # label = event['labels']['classification'].detach().numpy().flatten()
            #     # label[label > 0.5] = 1
            #     # label[label <= 0.5] = 0
            #     # pred = result_cu.detach().cpu().numpy().flatten()


            #     # m_loss = loss.detach().cpu().numpy()
            #     # m_accuracy = metrics.accuracy_score(label, pred)
            #     # m_f1 = metrics.f1_score(label, pred, average='binary')
            #     # m_precision = metrics.precision_score(label, pred, average='binary', zero_division=0)
            #     # m_recall = metrics.recall_score(label, pred, average='binary')



            # 

        if event['name'] == 'epoch_end':
            self.save()
