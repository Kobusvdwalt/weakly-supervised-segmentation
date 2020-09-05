from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
import numpy as np
import cv2
import albumentations
import os

from data.voc2012 import class_list_classification
from data.voc2012 import ImageToLabel

def composeAugmentation(source, size=256):
    if source == 'train':
        augmentation = albumentations.Compose(
        [
            albumentations.Rotate(5, always_apply=True),
            albumentations.LongestMaxSize(size, always_apply=True),
            albumentations.PadIfNeeded(size, size, cv2.BORDER_CONSTANT, 0),
            albumentations.HorizontalFlip(),
            albumentations.GaussNoise(),
            albumentations.Normalize(always_apply=True)
        ])
    else:
        augmentation = albumentations.Compose(
        [
            albumentations.LongestMaxSize(size, always_apply=True),
            albumentations.PadIfNeeded(size, size, cv2.BORDER_CONSTANT, 0),
            albumentations.Normalize(always_apply=True)
        ])

    return augmentation

class PascalVOCClassificationMulticlass(Dataset):
    def __init__(self, source='train'):
        self.classList = class_list_classification
        self.classCount = len(self.classList)
        
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'processed', 'classification_multiclass_' + source + '.txt')

        f = open(path, 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.source = source
        self.augmentation = composeAugmentation(source)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = idx
        parts = self.labels[sample].replace('\n', '').split(' ')
        
        # Image
        imageName = parts[0]
        image = cv2.imread('../VOC2012/JPEGImages/' + imageName + '.jpg')
        augmented = self.augmentation(image=image)
        image = augmented['image']

        # Label
        label = np.zeros(shape=(self.classCount))
        labelParts = parts[1].split('|')

        for lpart in labelParts:
            if lpart in self.classList:
                label[self.classList.index(lpart)] = 1
        
        # Label smoothing
        # https://arxiv.org/pdf/1906.02629.pdf
        label[label == 0] = 0.1
        label[label == 1] = 0.9
        return (image, label, imageName)

class PascalVOCClassificationBinary(Dataset):
    def __init__(self, source='train', target='aeroplane'):
        self.classList = class_list_classification
        self.classCount = len(self.classList)
        
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'processed', 'classification_binary_' + target + '_' + source + '.txt')

        f = open(path, 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.labels_true = []
        self.labels_false = []
        
        for sample in range(0, self.total-1):
            parts = self.labels[sample].replace('\n', '').split(' ')
            if (parts[1] == '1'):
                self.labels_true.append(sample)
            else:
                self.labels_false.append(sample)

        self.source = source
        self.augmentation = composeAugmentation(source)
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if (self.source == 'train'):
            if (random.randint(0, 100) > 50):
                sample = self.labels_true[random.randint(0, len(self.labels_true)-1)]
            else:
                sample = self.labels_false[random.randint(0, len(self.labels_false)-1)]
        else:
            sample = idx

        parts = self.labels[sample].replace('\n', '').split(' ')

        # Image
        imageName = parts[0]
        image = cv2.imread('../VOC2012/JPEGImages/' + imageName + '.jpg')

        augmented = self.augmentation(image=image)
        image = augmented['image']

        # Label
        label = np.zeros(shape=(2))

        if (parts[1] == '1'):
            label[0] = 1
            label[1] = 0
        else:
            label[0] = 0
            label[1] = 1

        # Label smoothing
        # https://arxiv.org/pdf/1906.02629.pdf
        label[label == 0] = 0.1
        label[label == 1] = 0.9
        return (image, label, imageName)

class PascalVOCSegmentation(Dataset):
    def __init__(self, source='train'):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'processed', 'segmentation_' + source + '.txt')
        f = open(path, 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.source = source

        self.augmentation = composeAugmentation(source)
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = idx

        # Read images and perform augmentation
        image_name = self.labels[sample].replace('\n', '')
        image = cv2.imread('../VOC2012/JPEGImages/' + image_name + '.jpg')
        
        image_width = image.shape[1]
        image_height = image.shape[0]

        if self.source == 'test':
            label = np.zeros(image.shape)
        else:
            label = cv2.imread('../VOC2012/SegmentationClass/' + image_name + '.png')

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask']

        # Construct Label        
        label_array = ImageToLabel(label)
        label = label_array

        return (image, label, image_name, image_width, image_height)
