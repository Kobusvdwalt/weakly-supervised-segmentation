import albumentations
import numpy as np
import cv2

class_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_count = len(class_list)

def get_augmentation(source, size=256):
    if source == 'train':
        augmentation = albumentations.Compose(
        [
            albumentations.ShiftScaleRotate(rotate_limit=15, always_apply=True),
            albumentations.Blur(blur_limit=5),
            albumentations.LongestMaxSize(size, always_apply=True),
            albumentations.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            albumentations.RandomBrightnessContrast(),
            albumentations.HorizontalFlip(),
            albumentations.Normalize(always_apply=True)
        ])
    else:
        augmentation = albumentations.Compose(
        [
            albumentations.LongestMaxSize(size, always_apply=True),
            albumentations.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            albumentations.Normalize(always_apply=True)
        ])

    return augmentation

def get_class_count():
    return len(class_list)

# Generates the color mappings for PASCAL_VOC
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

class_color_list = color_map()

# Input : 3 dim tensor (x, y, class_count)
# Output: RGB image (x, y, 3)
def label_to_image(label):
    image = np.zeros((label.shape[1], label.shape[2], 3))

    mask_a = np.argmax(label, 0)
    image[:, :, 0] = class_color_list[mask_a, 2] / 255
    image[:, :, 1] = class_color_list[mask_a, 1] / 255
    image[:, :, 2] = class_color_list[mask_a, 0] / 255

    return image

# Input : RGB image (x, y, 3)
# Output: 3 dim tensor (class_count, x, y)
def image_to_label(image):
    label = np.zeros((class_count, image.shape[0], image.shape[1]))
    for i in range(0, class_count):
        feature = np.all(image == (class_color_list[i, 2], class_color_list[i, 1], class_color_list[i, 0]), axis=-1)
        label[i] = feature[:, :] / 1.0

    return label

# Input : RGB image (x, y, 3)
# Output: 1 dim vector of (class_count)
def image_to_classes(image):
    classifcation_label = np.zeros(class_count)
    for i in range(0, class_count):
        feature = np.all(image == (class_color_list[i, 2], class_color_list[i, 1], class_color_list[i, 0]), axis=-1)

        if np.sum(feature) > 0:
            classifcation_label[i] = 1

    return classifcation_label


# Input : 3 dim tensor (x, y, class_count)
# Output: 1 dim vector of (class_count)
def label_to_classes(label):
    classifcation_label = np.zeros(class_count)
    for i in range(0, class_count):
        if np.sum(label[i]) > 0:
            classifcation_label[i] = 1

    return classifcation_label

# Input : One-hot encoded vector
# Output: List of corresponding class descriptions
def classes_to_words(classes):
    words = []
    for i in range(0, class_count):
        if (classes[i] == 1):
            words.append(class_list[i])

    return words

def label_smoothing(inputs):
    # Label smoothing
    # https://arxiv.org/pdf/1906.02629.pdf
    inputs[inputs == 0] = 0.01
    inputs[inputs == 1] = 1 - 0.01

    return inputs