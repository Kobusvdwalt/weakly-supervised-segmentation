
import cv2
import torch
import numpy as np
import wandb

from models._common import ModelBase, build_vgg_features
import torch.sparse as sparse

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

        self.features = build_vgg_features(pretrained=True, unfreeze_from=2)
        self.features[4] = torch.nn.MaxPool2d(3, 2, 1)
        self.features[9] = torch.nn.MaxPool2d(3, 2, 1)
        self.features[16] = torch.nn.MaxPool2d(3, 2, 1)
        self.features[23] = torch.nn.MaxPool2d(3, 1, 1)
        self.features[28] = torch.nn.Conv2d(512, 448, 1)
        
        self.upsample = torch.nn.Upsample(scale_factor=16, mode='bilinear')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0.0001)

        self.predefined_featuresize = int(448//8)
        self.ind_from, self.ind_to = get_indices_of_pairs(5, (self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)

        self.comp_image = None

    def get_affinity(self, image, dense=False):
        x = self.features(image)

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = get_indices_of_pairs(5, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from)
            ind_to = torch.from_numpy(ind_to)
        
        x = x.view(x.size(0), x.size(1), -1)

        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
        
        aff = torch.abs(ft-ff)
        aff = torch.mean(aff, dim=1)
        aff = torch.exp(-aff)

        if dense:
            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1), torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()
            return aff_mat
        else:
            return aff

    def event(self, event):
        super().event(event)

        if event['name'] == 'infer_aff_net_dense':
            image_cu = event['image'].cuda(non_blocking=True)
            return self.get_affinity(image_cu, True)

        if event['name'] == 'minibatch':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_affinity_bg_cu = event['labels']['affinity'][0].cuda(non_blocking=True)
            label_affinity_fg_cu = event['labels']['affinity'][1].cuda(non_blocking=True)
            label_affinity_ng_cu = event['labels']['affinity'][2].cuda(non_blocking=True)

            if self.comp_image is None:
                self.comp_image = image_cu.clone().detach().cpu()
            
            result_cu = self.get_affinity(image_cu)

            bg_count = torch.sum(label_affinity_bg_cu) + 1e-5
            fg_count = torch.sum(label_affinity_fg_cu) + 1e-5
            neg_count = torch.sum(label_affinity_ng_cu) + 1e-5

            bg_loss = torch.sum(- label_affinity_bg_cu * torch.log(result_cu + 1e-5)) / bg_count
            fg_loss = torch.sum(- label_affinity_fg_cu * torch.log(result_cu + 1e-5)) / fg_count
            neg_loss = torch.sum(- label_affinity_ng_cu * torch.log(1. + 1e-5 - result_cu)) / neg_count

            loss = bg_loss/4 + fg_loss/4 + neg_loss/2

            wandb.log({
                'loss': loss,
                'loss_bg': bg_loss,
                'loss_fg': fg_loss,
                'loss_ne': neg_loss,
            }, step=event['step'])

            if self.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(event['step'], end='\r')

            # Print metrics
            if event['batch'] % 100 == 0:
                with torch.no_grad():
                    def get_aff_sum(aff_in):
                        aff_lab = np.sum(aff_in, axis=0) / aff_in.shape[0]
                        aff_lab = np.reshape(aff_lab, (52, 48))
                        aff_lab = cv2.resize(aff_lab, (448, 448), interpolation=cv2.INTER_NEAREST)
                        return aff_lab

                    comp_image_in = self.comp_image.clone().cuda(non_blocking=True)
                    pred = self.get_affinity(comp_image_in)

                    for i in range(pred.shape[0]):
                        wandb.log({
                            'img_' + str(i) : wandb.Image(np.moveaxis(comp_image_in[i].detach().cpu().numpy(), 0, -1)),
                            'pred_' + str(i): wandb.Image(get_aff_sum(pred[i].detach().cpu().numpy())),
                        }, step=event['step'])

        if event['name'] == 'epoch_end':
            print('')
            self.save()

