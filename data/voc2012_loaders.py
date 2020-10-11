from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations
import os

from data.voc2012 import class_list
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

class PascalVOCClassification(Dataset):
    def __init__(self, source='train'):
        self.classList = class_list[1:]
        self.classCount = len(self.classList)
        
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'output', 'voc2012_classification_' + source + '.txt')

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
        image_name = parts[0]
        image = cv2.imread('../datasets/voc2012/JPEGImages/' + image_name + '.jpg')
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

        meta = np.array([256, 256, 0, 0])

        return (image, label, image_name, meta)

class PascalVOCSegmentation(Dataset):
    def __init__(self, source='train'):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'output', 'voc2012_segmentation_' + source + '.txt')
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
        image = cv2.imread('../datasets/voc2012/JPEGImages/' + image_name + '.jpg')
        
        image_width = image.shape[1]
        image_height = image.shape[0]

        if self.source == 'test':
            label = np.zeros(image.shape)
        else:
            label = cv2.imread('../datasets/voc2012/SegmentationClass/' + image_name + '.png')

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask']

        # Construct Label        
        label_array = ImageToLabel(label)
        label = label_array

        meta = np.array([256, 256, image_width, image_height])

        return (image, label, image_name, meta)


class PascalVOCSelfsupervised(Dataset):
    def __init__(self, source='train'):
        self.classList = class_list[1:]
        self.classCount = len(self.classList)

        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'output', 'voc2012_classification_' + source + '.txt')
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
        image_name = self.labels[sample].replace('\n', '').split(' ')[0]

        image = cv2.imread('../datasets/voc2012/JPEGImages/' + image_name + '.jpg')
        label = cv2.imread('../datasets/voc2012/JPEGImages/' + image_name + '.jpg')

        image_width = image.shape[1]
        image_height = image.shape[0]

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask'] / 255.0


        classification_label = np.ones(shape=(20)) * 0.9
        classification_label_parts = self.labels[sample].replace('\n', '').split(' ')[1].split('|')

        for word in self.classList:
            if word not in classification_label_parts:
                classification_label[self.classList.index(word)] = 0.1


        # classification_label = classification_label[:, np.newaxis, np.newaxis]

        meta = np.array([256, 256, image_width, image_height])

        return (image, label, image_name, meta, classification_label)

