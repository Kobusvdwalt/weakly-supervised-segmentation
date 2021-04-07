from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations

from data.voc2012 import image_to_label, label_to_classes

def composeAugmentation(source, size=256):
    if source == 'train':
        augmentation = albumentations.Compose(
        [
            albumentations.ShiftScaleRotate(rotate_limit=15, always_apply=True),
            albumentations.Blur(blur_limit=5),
            albumentations.LongestMaxSize(size, always_apply=True),
            albumentations.PadIfNeeded(size, size, cv2.BORDER_CONSTANT, 0),
            albumentations.RandomBrightnessContrast(),
            albumentations.HorizontalFlip(),
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

def segmentation_labels(source):
    f = open('datasets/voc2012/ImageSets/Segmentation/' + source + '.txt', 'r')
    lines = f.readlines()

    # Return the data as two arrays for easy access
    images_array = []
    labels_array = []
    for image_name in lines:
        images_array.append(image_name.replace('\n', ''))
        labels_array.append(image_name.replace('\n', ''))

    return images_array, labels_array

class PascalVOCSegmentation(Dataset):
    def __init__(self, source='train'):
        images, labels = segmentation_labels(source)
        self.source = source
        self.images = images
        self.labels = labels
        self.total = len(self.labels)
        self.augmentation = composeAugmentation(source)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample = idx

        # Read images and perform augmentation
        image_name = self.labels[sample]
        image = cv2.imread('datasets/voc2012/JPEGImages/' + image_name + '.jpg')
        
        image_width = image.shape[1]
        image_height = image.shape[0]

        if self.source == 'test':
            label = np.zeros(image.shape)
        else:
            label = cv2.imread('datasets/voc2012/SegmentationClass/' + image_name + '.png')

        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask']

        # Construct Label
        label = image_to_label(label)

        inputs = {
            'image': np.moveaxis(image, 2, 0),
            'label': np.delete(label_to_classes(label), 0)
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
