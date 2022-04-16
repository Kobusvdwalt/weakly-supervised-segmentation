
import cv2
import torch
import numpy as np

from data.voc2012 import label_to_image
from models._common import ModelBase, build_vgg_features
from training._common import move_to
import torch.nn.functional as F


def get_indices_of_pairs(radius, size):

    search_dist = []

    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])
        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to

class Vgg16Aff(ModelBase):
    def __init__(self, class_count=20, **kwargs):
        super(Vgg16Aff, self).__init__(**kwargs)
        self.class_count = class_count

        self.features = build_vgg_features(pretrained=True, unfreeze_from=10)
        self.features[4] = torch.nn.MaxPool2d(3, 2, 1)
        self.features[9] = torch.nn.MaxPool2d(3, 2, 1)
        self.features[16] = torch.nn.MaxPool2d(3, 2, 1)
        self.features[23] = torch.nn.MaxPool2d(3, 1, 1)
        
        self.features[28] = torch.nn.Conv2d(512, 448, 1)
        print(self.features)
        
        # self.drop_3 = torch.nn.Dropout2d(p=0.5)
        # self.conv_3 = torch.nn.Conv2d(512, self.class_count, 1, bias=False)
        # self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.flat = torch.nn.Flatten(1, 3)
        # self.relu = torch.nn.LeakyReLU()
        # self.sigmoid = torch.nn.Sigmoid()

        self.upsample = torch.nn.Upsample(scale_factor=16, mode='bilinear')
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.timer = None

        self.predefined_featuresize = int(448//8)
        self.ind_from, self.ind_to = get_indices_of_pairs(5, (self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)

    def get_affinity(self, image):
        x = self.features(image)
        
        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        # else:
        #     ind_from, ind_to = pyutils.get_indices_of_pairs(5, (x.size(2), x.size(3)))
        #     ind_from = torch.from_numpy(ind_from); ind_to = torch.from_numpy(ind_to)
        
        x = x.view(x.size(0), x.size(1), -1)

        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
        
        aff = torch.abs(ft-ff)
        aff = torch.mean(aff, dim=1)
        aff = torch.exp(-aff)

        return aff

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
        
        if event['name'] == 'infer_aff_mat_dense':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            result_cu = self.get_affinity(image_cu)
            return result_cu

        if event['name'] == 'minibatch':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_affinity_bg_cu = event['labels']['affinity'][0].cuda(non_blocking=True)
            label_affinity_fg_cu = event['labels']['affinity'][1].cuda(non_blocking=True)
            label_affinity_ng_cu = event['labels']['affinity'][2].cuda(non_blocking=True)

            def get_aff_sum(aff_in):
                aff_lab = np.sum(aff_in, axis=0) / aff_in.shape[0]
                aff_lab = np.reshape(aff_lab, (52, 48))
                aff_lab = cv2.resize(aff_lab, (448, 448), interpolation=cv2.INTER_NEAREST)
                return aff_lab
            
            img = np.moveaxis(image_cu[0].detach().cpu().numpy(), 0, -1)
            cv2.imshow('img', img)
            cv2.imshow('aff_fg', get_aff_sum(label_affinity_fg_cu[0].detach().cpu().numpy()))
            cv2.imshow('aff_bg', get_aff_sum(label_affinity_bg_cu[0].detach().cpu().numpy()))
            cv2.imshow('aff_neg', get_aff_sum(label_affinity_ng_cu[0].detach().cpu().numpy()))
            cv2.waitKey(1)

            result_cu = self.get_affinity(image_cu)

            bg_count = torch.sum(label_affinity_bg_cu) + 1e-5
            fg_count = torch.sum(label_affinity_fg_cu) + 1e-5
            neg_count = torch.sum(label_affinity_ng_cu) + 1e-5

            bg_loss = torch.sum(- label_affinity_bg_cu * torch.log(result_cu + 1e-5)) / bg_count
            fg_loss = torch.sum(- label_affinity_fg_cu * torch.log(result_cu + 1e-5)) / fg_count
            neg_loss = torch.sum(- label_affinity_ng_cu * torch.log(1. + 1e-5 - result_cu)) / neg_count

            loss = bg_loss/4 + fg_loss/4 + neg_loss/2

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
