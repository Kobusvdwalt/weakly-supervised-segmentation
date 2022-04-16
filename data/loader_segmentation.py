import os
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from data.affinity_label import ExtractAffinityLabelInRadius
from data.voc2012 import image_to_label, label_to_classes, get_augmentation, label_smoothing

def read_file(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    lines_formatted = []
    for line in lines:
        lines_formatted.append(line.replace('\n', ''))

    return lines_formatted

def read_cam(file_path, width, height):
    cam = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    cam = np.reshape(cam, (cam.shape[0] // height, height, width))
    cam = np.moveaxis(cam, 0, -1)
    return cam

class Segmentation(Dataset):
    def __init__(self,
        dataset_root,
        source = 'train',
        source_augmentation='val',
        image_size = 256,
        requested_labels = ['classification'],
        affinity_root=None,
    ):
        self.dataset_root = dataset_root
        self.image_root = dataset_root + '/images'
        self.label_root = dataset_root + '/labels'

        images = read_file(dataset_root + '/' + source + '.txt')
        labels = images

        self.source = source
        self.images = images
        self.labels = labels
        self.requested_labels = requested_labels
        self.total = len(self.labels)
        self.augmentation = get_augmentation(source_augmentation, image_size=image_size)
        self.affinity_root = affinity_root

        if self.affinity_root:
            self.affinity_label = ExtractAffinityLabelInRadius(image_size//8, radius=5)

    def __len__(self):
        return self.total

    def __getitem__(self, sample):
        # Read images and perform augmentation
        image_name = self.labels[sample]
        image_path = os.path.join(self.image_root, image_name + '.jpg')
        image = cv2.imread(image_path)

        image_width = image.shape[1]
        image_height = image.shape[0]

        # Read masks
        masks = []

        # - Read ground truth semgentation mask
        label_path = os.path.join(self.label_root, image_name + '.png')
        label = cv2.imread(label_path)
        masks.append(label)

        # - Read affinity masks
        if 'affinity' in self.requested_labels:
            img_la = read_cam(os.path.join(self.affinity_root, 'cam_la', image_name + '.png'), image_width, image_height)
            img_ha = read_cam(os.path.join(self.affinity_root, 'cam_ha', image_name + '.png'), image_width, image_height)

            masks.append(img_la)
            masks.append(img_ha)

        # Apply augmentations
        transform = self.augmentation(image=image, masks=masks)
        image = transform['image']
        image_augmented_height = image.shape[0]
        image_augmented_width = image.shape[1]
        image = np.moveaxis(image, -1, 0)

        # Build inputs
        inputs = {
            'image': image,
        }

        # Build labels
        labels = {}
        for requested_label in self.requested_labels:
            if requested_label == 'classification':
                segmentation = image_to_label(transform['masks'][0])
                classification = np.delete(label_to_classes(segmentation), 0).astype(np.float32)
                labels[requested_label] = label_smoothing(classification)
            if requested_label == 'segmentation':
                segmentation = image_to_label(transform['masks'][0])
                labels[requested_label] = label_smoothing(segmentation).astype(np.float32)
            if requested_label == 'affinity':
                labels[requested_label] = self.affinity_label.get(transform['masks'][1], transform['masks'][2])

        # Build meta data
        data_package = {
            'image_path': image_path,
            'image_name': image_name,
            'label_path': label_path,
            'width': image_width,
            'height': image_height,
            'augmented_width': image_augmented_width,
            'augmented_height': image_augmented_height,
        }
        
        return (inputs, labels, data_package)
