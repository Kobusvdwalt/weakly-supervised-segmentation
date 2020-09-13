from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
import numpy as np
import cv2
import albumentations

classListArray = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class PascalVOCClassificationBinary(Dataset):
    def composeAugmentation(self):
        if self.source == 'train':
            self.augment = albumentations.Compose(
            [
                albumentations.LongestMaxSize(256, always_apply=True),
                albumentations.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, 0),
                albumentations.HorizontalFlip(),
            ])
        else:
            self.augment = albumentations.Compose(
            [
                albumentations.LongestMaxSize(256, always_apply=True),
                albumentations.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, 0),
            ])

    def __init__(self, classLabel, source='train', random=False):
        self.classList = classListArray
        self.classCount = len(self.classList)
        
        dir = __file__.split('\\')
        dir.pop()
        dir = '\\'.join(dir)

        self.labels = {}
        self.total = 0
        for cl in classListArray:
            f = open(dir + '\\data\\'+ cl + '_' + source + '.txt', 'r')
            self.labels[cl] = f.readlines()
            self.total += len(self.labels[cl])
            f.close()

        self.source = source
        self.classLabel = classLabel
        self.random = random

        self.sizeX = 256
        self.sizeY = 256

        self.composeAugmentation()
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        selectedClass = self.classLabel
        if (random.randint(0, 100) > 50):
            selectedClass = classListArray[random.randint(0, len(classListArray) -1)]

        selectedImage = self.labels[selectedClass][random.randint(0, len(self.labels[selectedClass]) -1)]
        selectedImage = selectedImage.replace('\n', '')

        # Image
        image = cv2.imread('../voc2012/JPEGImages/' + selectedImage + '.jpg')

        et = self.augment(image=image)
        image = et['image'] / 255.0

        # Label
        label = np.zeros(shape=(1))
        if (self.classLabel == selectedClass):
            label[0] = 1

        return (image, label)