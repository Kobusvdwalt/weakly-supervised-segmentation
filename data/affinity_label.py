import os
import cv2
import numpy as np
import torch

def avg_pool_2d(img, ksize):
    import skimage.measure
    return skimage.measure.block_reduce(img, (ksize, ksize, 1), np.mean)

class ExtractAffinityLabelInRadius():
    def __init__(self, cropsize, radius=5):
        self.radius = radius
        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def get(self, mask_la, mask_ha):
        # Pool
        resize_scale = 8
        mask_la = avg_pool_2d(mask_la, resize_scale)
        mask_ha = avg_pool_2d(mask_ha, resize_scale)

        # Move channels
        mask_la = np.moveaxis(mask_la, -1, 0)
        mask_ha = np.moveaxis(mask_ha, -1, 0)

        # Argmax over channels
        label_la = np.argmax(mask_la, axis=0).astype(np.uint8)
        label_ha = np.argmax(mask_ha, axis=0).astype(np.uint8)

        # TODO: Still don't fully understand this but it combines the la/ha labels
        no_score_region = np.max(np.concatenate([label_la, label_ha]), 0) < 0.01
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255
        
        label = self.process(label)

        return label
        

    def process(self, label):
        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)
