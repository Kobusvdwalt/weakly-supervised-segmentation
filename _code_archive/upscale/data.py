from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
import numpy as np
import cv2
import albumentations

classListArray = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
classCount = len(classListArray)
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

color_map_c = color_map()

class PascalVOCSegmentation(Dataset):
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

    def __init__(self, source='train', random=False, returnOriginal=False):
        f = open('data/seg_' + source + '.txt', 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.source = source
        self.random = random
        self.returnOriginal = returnOriginal

        self.sizeX = 256
        self.sizeY = 256

        self.composeAugmentation()
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if (self.random == True):
            sample = random.randint(0, self.total-1)
        else:
            sample = idx

        # Read images and perform augmentation
        imageName = self.labels[sample].replace('\n', '')
        image = cv2.imread('../VOC2012/JPEGImages/' + imageName + '.jpg')
        label = cv2.imread('../VOC2012/SegmentationClass/' + imageName + '.png')
        transform = self.augment(image=image, mask=label)
        image = transform['image'] / 255.0
        label = transform['mask']

        # Construct Label        
        label_array = np.zeros((classCount, 256, 256))
        valid_list = []
        for i in range(1, classCount):
            feature = np.all(label == (color_map_c[i, 2], color_map_c[i, 1], color_map_c[i, 0]), axis=-1)
            label_array[i] = feature[:, :] / 1.0
            if np.sum(label_array[i]) > 0:
                valid_list.append(i)
                break

        index = random.randint(0, len(valid_list)-1)
        label_selected = label_array[valid_list[index]]

        # Resize
        #label_image_tiny_r = label_selected
        label_image_tiny = cv2.resize(label_selected, (16, 16))
        label_image_tiny_r = cv2.resize(label_image_tiny, (256, 256), None, 0, 0, cv2.INTER_NEAREST)
        label_image_micro = cv2.resize(label_selected, (8, 8))
        label_image_micro_r = cv2.resize(label_image_micro, (256, 256), None, 0, 0, cv2.INTER_LINEAR)

        target = label_image_tiny_r[:, :, np.newaxis]
        original = label_selected[:, :, np.newaxis]

        inputs = np.zeros((self.sizeX, self.sizeY, 4))
        inputs[:, :, :3] = image
        inputs[:, :, 3] = label_image_micro_r

        if self.returnOriginal:
            return (inputs, target, original)
        else:
            return (inputs, target)
