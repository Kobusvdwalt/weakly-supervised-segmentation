from data.voc2012 import ClassesToWords
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations

from data.voc2012_loader_classification import PascalVOCClassification

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

class PascalVOCSelfsupervised(Dataset):
    def __init__(self, source='train'):
        self.loader_classification = PascalVOCClassification(source)
        self.augmentation = composeAugmentation(source)
    def __len__(self):
        return self.loader_classification.total

    def __getitem__(self, idx):
        # Get Image and Label
        image_name, image = self.loader_classification.get_image_raw(idx)        
        image_name, label = self.loader_classification.get_image_raw(idx)
        
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Augment image
        transform = self.augmentation(image=image, mask=label)
        image = transform['image']
        label = transform['mask'] / 255.0

        # Get classification label
        classification_label = self.loader_classification.get_label_raw(idx)

        # Debug image and label
        # cv2.imshow('input', image)
        # maxed = classification_label
        # maxed[maxed > 0.5] = 1
        # maxed[maxed < 0.5] = 0
        # maxed = np.insert(maxed, 0, 0)
        # words = ClassesToWords(maxed)
        
        # print(words)
        # cv2.waitKey(0)

        inputs = {
            'image': np.moveaxis(image, 2, 0)
        }

        labels = {
            # 'classification': classification_label,
            'reconstruction': np.moveaxis(label, 2, 0)  
        }
        data_package = {
            'image_name': image_name,
            'width': image_width,
            'height': image_height,
            'augmented_width': 256,
            'augmented_height': 256,
            'classification_label': classification_label
        }

        return (inputs, labels, data_package)