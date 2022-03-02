import os
from torch.utils.data import Dataset
import numpy as np
import cv2

from data.voc2012 import image_to_label, label_to_classes, get_augmentation, label_smoothing

def read_file(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    lines_formatted = []
    for line in lines:
        lines_formatted.append(line.replace('\n', ''))

    return lines_formatted

class Segmentation(Dataset):
    def __init__(self,
        dataset_root,
        source = 'train',
        image_size = 256
    ):
        self.dataset_root = dataset_root
        self.image_root = dataset_root + '/images'
        self.label_root = dataset_root + '/labels'

        images = read_file(dataset_root + '/' + source + '.txt')
        labels = images

        self.source = source
        self.images = images
        self.labels = labels
        self.total = len(self.labels)
        self.augmentation = get_augmentation(source, image_size=image_size)

    def __len__(self):
        return self.total

    def __getitem__(self, sample):
        # Read images and perform augmentation
        image_name = self.labels[sample]
        image_path = os.path.join(self.image_root, image_name + '.jpg')
        image = cv2.imread(image_path)

        label_path = os.path.join(self.label_root, image_name + '.png')
        label = cv2.imread(label_path)

        image_width = image.shape[1]
        image_height = image.shape[0]

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask']

        image = np.moveaxis(image, 2, 0)
        segmentation = image_to_label(label)
        classification = np.delete(label_to_classes(segmentation), 0)

        inputs = {
            'image': image,
        }

        labels = {
            'segmentation': label_smoothing(segmentation),
            'segmentation_cat':  np.argmax(segmentation, axis=0).astype(np.int64),
            'classification': label_smoothing(classification).astype(np.float32)
        }

        data_package = {
            'image_path': image_path,
            'image_name': image_name,
            'width': image_width,
            'height': image_height,
            'augmented_width': 256,
            'augmented_height': 256,
        }

        return (inputs, labels, data_package)
