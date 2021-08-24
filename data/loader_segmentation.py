from torch.utils.data import Dataset
from random import Random
import numpy as np
import cv2

from data.voc2012 import image_to_label, label_to_classes, get_augmentation, label_smoothing, destroy_shape

def read_file(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    lines_formatted = []
    for line in lines:
        lines_formatted.append(line.replace('\n', ''))

    return lines_formatted

class VOCSegmentation(Dataset):
    def __init__(self, source='train', dataset='voc'):
        if dataset == 'voc':
            dataset_root = 'datasets/voc2012/ImageSets/Segmentation/'
            self.image_root = 'datasets/voc2012/JPEGImages/'
            self.label_root = 'datasets/voc2012/SegmentationClass/'
        if dataset == 'voco':
            dataset_root = 'datasets/voco/'
            self.image_root = 'datasets/voco/images/'
            self.label_root = 'datasets/voco/labels/'

        images = read_file(dataset_root + source + '.txt')
        labels = images

        self.source = source
        self.images = images
        self.labels = labels
        self.total = len(self.labels)
        self.augmentation = get_augmentation(source)

    def __len__(self):
        return self.total

    def __getitem__(self, sample):
        # Read images and perform augmentation
        image_name = self.labels[sample]
        image = cv2.imread(self.image_root + image_name + '.jpg')
        label = cv2.imread(self.label_root + image_name + '.png')

        image_width = image.shape[1]
        image_height = image.shape[0]

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask']

        image = np.moveaxis(image, 2, 0)
        label = image_to_label(label)
        classification = np.delete(label_to_classes(label), 0)

        inputs = {
            'image': image,
        }

        labels = {
            'segmentation': label_smoothing(label),
            'classification': label_smoothing(classification)
        }

        data_package = {
            'image_name': image_name,
            'width': image_width,
            'height': image_height,
            'augmented_width': 256,
            'augmented_height': 256,
        }

        return (inputs, labels, data_package)
