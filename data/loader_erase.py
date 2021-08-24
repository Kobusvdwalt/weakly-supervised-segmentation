from torch.utils.data import Dataset
from random import Random
import numpy as np
import cv2

from data.voc2012 import image_to_label, label_to_classes, get_augmentation, label_smoothing, destroy_shape, downsample_shape

def read_file(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    lines_formatted = []
    for line in lines:
        lines_formatted.append(line.replace('\n', ''))

    return lines_formatted

class VOCErase(Dataset):
    def __init__(self, source='train', type='none', size=0, dataset='voc'):
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

        self.type = type
        self.size = size
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

        # Construct erase map
        if self.type == 'none':
            pass

        if self.type == 'erase_downsample':
            erase_map = np.max(label[1:], axis=0)
            erase_map = downsample_shape(erase_map, self.size)
            image[:, erase_map > 0.5] = 0.5

        if self.type == 'erase_bbox':
            # Results in a RGB image where the objects of interesed are covered by grey bounding boxes
            for l in label[1:]:
                mask = np.zeros((image.shape[1], image.shape[2]))
                mask[l > 0.5] = 1
                mask[l <= 0.5] = 0

                points = cv2.findNonZero(mask)
                if points is None:
                    continue
                (x, y, w, h) = cv2.boundingRect(points)
                
                mask = np.zeros((image.shape[1], image.shape[2]))
                mask = cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y+h)), 1, thickness=-1)
                image[:, mask > 0.5] = 0.5

        if self.type == 'erase_bbnc':
            image = np.ones(image.shape) * 0.05
            for l in label[1:]:
                mask = np.zeros((image.shape[1], image.shape[2]))
                mask[l > 0.5] = 1
                mask[l <= 0.5] = 0

                points = cv2.findNonZero(mask)
                if points is None:
                    continue
                (x, y, w, h) = cv2.boundingRect(points)
                
                mask = np.zeros((image.shape[1], image.shape[2]))
                mask = cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y+h)), 1, thickness=-1)
                image[:, mask > 0.5] = 0.5

        if self.type == 'erase_gaus':
            erase_map = np.max(label[1:], axis=0)
            erase_map = destroy_shape(erase_map, self.size)
            image[:, erase_map > 0.5] = 0.5

        if self.type == 'mask_base':
            image = np.ones(image.shape) * 0.01
            rng = Random(sample)
            for l in label[1:]:
                m = destroy_shape(l, self.size)
                image[:, m > 0.5] = rng.random()

        if self.type == 'mask_random_bg':
            image = np.ones(image.shape) * 0.5
            rng = Random(sample)
            for l in label[1:]:
                image_random = cv2.imread('datasets/voc2012/JPEGImages/' + self.labels[rng.randint(0, len(self.labels)-1)] + '.jpg')
                transform_random = self.augmentation(image=image_random)
                image_random = transform_random['image']
                image_random = np.moveaxis(image_random, 2, 0)

                m = destroy_shape(l, self.size)
                image[:, m > 0.5] = image_random[:, m > 0.5]

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
