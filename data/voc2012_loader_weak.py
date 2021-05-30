from torch.utils.data import Dataset
import numpy as np
import cv2, os

from data.voc2012 import image_to_label, get_augmentation

class PascalVOCSegmentationWeak(Dataset):
    def __init__(self, source='train', vis_folder=''):
        self.vis_folder = vis_folder
        dir_list = os.listdir(vis_folder)
        self.labels = []

        for dir in dir_list:
            parts = dir.split('.')
            file = parts[0]
            ext = parts[1]
            if ext == 'png':
                self.labels.append(file)

        self.total = len(self.labels)
        self.source = source
        self.augmentation = get_augmentation(source)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = idx
        image_name = self.labels[sample]
        image = cv2.imread('datasets/voc2012/JPEGImages/' + image_name + '.jpg')
        label = cv2.imread(self.vis_folder + image_name + '.png')

        image_width = image.shape[1]
        image_height = image.shape[0]

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask']

        # Construct Label
        label = image_to_label(label)

        inputs = {
            'image': np.moveaxis(image, 2, 0)
        }

        labels = {
            'segmentation': label
        }

        data_package = {
            'image_name': image_name,
            'width': image_width,
            'height': image_height,
            'augmented_width': 256,
            'augmented_height': 256,
        }

        return (inputs, labels, data_package)