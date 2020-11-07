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

        data_package = {
            'image_name': image_name,
            'width': image_width,
            'height': image_height,
            'augmented_width': 256,
            'augmented_height': 256,
            'classification_label': classification_label
        }

        image = np.moveaxis(image, 2, 0)
        label = np.moveaxis(label, 2, 0)

        return (image, label, data_package)