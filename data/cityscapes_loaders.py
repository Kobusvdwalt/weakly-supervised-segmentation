from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations
import sys, os
sys.path.insert(0, os.path.abspath('../'))

from data.cityscapes import get_label_words, image_to_label

def compose_augmentation(source, size=256):
    if source == 'train':
        augmentation = albumentations.Compose(
        [
            # albumentations.Rotate(5, always_apply=True),
            albumentations.LongestMaxSize(512, always_apply=True),
            # albumentations.PadIfNeeded(size, size, cv2.BORDER_CONSTANT, 0),
            # albumentations.HorizontalFlip(),
            # albumentations.GaussNoise(),
            albumentations.Normalize(always_apply=True)
        ])
    else:
        augmentation = albumentations.Compose(
        [
            albumentations.LongestMaxSize(512, always_apply=True),
            albumentations.Normalize(always_apply=True)
        ])

    return augmentation

class CityscapesClassification(Dataset):
    def __init__(self, source='train'):
        self.classList = get_label_words()[1:]
        self.classCount = len(self.classList)
        
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'output', 'cityscapes_classification_' + source + '.txt')

        f = open(path, 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.source = source
        self.augmentation = compose_augmentation(source)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = idx
        parts = self.labels[sample].replace('\n', '').split(' ')
        
        # Image
        image_name = parts[0]
        image = cv2.imread('../datasets/cityscapes/' + image_name)

        augmented = self.augmentation(image=image)
        image = augmented['image']

        # Label
        label = np.zeros(shape=(self.classCount))
        label_parts = parts[1].split('|')

        for lpart in label_parts:
            if lpart in self.classList:
                label[self.classList.index(lpart)] = 1
        
        # Label smoothing
        # https://arxiv.org/pdf/1906.02629.pdf
        label[label == 0] = 0.1
        label[label == 1] = 0.9

        meta = np.array([256, 512, 0, 0])

        return (image, label, image_name, meta)

class CityscapesSegmentation(Dataset):
    def __init__(self, source='train'):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'output', 'cityscapes_segmentation_' + source + '.txt')
        f = open(path, 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.source = source

        self.augmentation = compose_augmentation(source)
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = idx

        # Read images and perform augmentation
        image_name = self.labels[sample].replace('\n', '').split(' ')[0]
        label_name = self.labels[sample].replace('\n', '').split(' ')[1]
        image = cv2.imread('../datasets/cityscapes/' + image_name)
        label = cv2.imread('../datasets/cityscapes/' + label_name)
        image_width = image.shape[1]
        image_height = image.shape[0]

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask']

        # Construct Label        
        label_array = image_to_label(label)
        label = label_array

        image_name = image_name.split('/')[-1].replace('.png', '')

        meta = np.array([256, 512, image_width, image_height])

        return (image, label, image_name, meta)

class CityscapesSelfsupervised(Dataset):
    def __init__(self, source='train'):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(package_directory, 'output', 'cityscapes_segmentation_' + source + '.txt')
        f = open(path, 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.source = source

        self.augmentation = compose_augmentation(source)
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = idx

        # Read images and perform augmentation
        image_name = self.labels[sample].replace('\n', '').split(' ')[0]
        image = cv2.imread('../datasets/cityscapes/' + image_name)
        label = cv2.imread('../datasets/cityscapes/' + image_name)

        image_width = image.shape[1]
        image_height = image.shape[0]

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask'] / 255.0

        meta = np.array([256, 256, image_width, image_height])

        return (image, label, image_name, meta)

