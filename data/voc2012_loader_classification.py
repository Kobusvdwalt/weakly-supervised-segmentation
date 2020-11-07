from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations

from data.voc2012 import class_list

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

def classification_labels(source, class_list_filtered):
    label_map = {}
    # Iterate over each selected class
    for class_name in class_list_filtered:
        # Open the label data for that class
        f = open('../datasets/voc2012/ImageSets/Main/' + class_name + '_' + source + '.txt', 'r')
        lines = f.readlines()

        # Iterate over each image
        for line in lines:
            line = line.replace('\n', '')
            parts = line.split(' ')

            image_name = parts[0]

            # Create empty label if image_name not in map
            if image_name not in label_map:
                label_map[image_name] = np.zeros(shape=(len(class_list_filtered)))

            # Assign the correct positive bit
            if (len(parts) > 2 and parts[2] == '1'):
                label_map[image_name][class_list_filtered.index(class_name)] = 1

    # Return the data as two arrays for easy access
    images_array = []
    labels_array = []

    for image_name, label in label_map.items():
        # Label smoothing
        # https://arxiv.org/pdf/1906.02629.pdf
        label[label == 0] = 0.1
        label[label == 1] = 0.9

        images_array.append(image_name)
        labels_array.append(label)        

    return images_array, labels_array

class PascalVOCClassification(Dataset):
    def __init__(self, source='train'):
        self.class_list = class_list[1:]
        self.class_count = len(self.class_list)

        images, labels = classification_labels(source, self.class_list)
        self.source = source
        self.images = images
        self.labels = labels
        self.total = len(self.labels)
        self.augmentation = composeAugmentation(source)

    def __len__(self):
        return self.total

    def get_image_raw(self, sample):
        image_name = self.images[sample]
        image = cv2.imread('../datasets/voc2012/JPEGImages/' + image_name + '.jpg')
        return image_name, image

    def get_label_raw(self, sample):
        return self.labels[sample]

    def __getitem__(self, idx):
        # Image
        image_name, image = self.get_image_raw(idx)
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Augment image
        augmented = self.augmentation(image=image)
        image = augmented['image']

        # Label
        label = self.get_label_raw(idx)

        data_package = {
            'image_name': image_name,
            'width': image_width,
            'height': image_height,
            'augmented_width': 256,
            'augmented_height': 256,
        }

        image = np.moveaxis(image, 2, 0)

        return (image, label, data_package)
