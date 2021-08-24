from torch.utils.data import Dataset
import numpy as np
import cv2

# Generic method to get image file names and remove the new line
def read_file(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    lines_formatted = []
    for line in lines:
        lines_formatted.append(line.replace('\n', ''))

    return lines_formatted

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

class_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_count = len(class_list)
class_color_list = color_map()

# Method to take RGB image and return N_Channel label
def image_to_label(image):
    label = np.zeros((class_count, image.shape[0], image.shape[1]))
    for i in range(0, class_count):
        feature = np.all(image == (class_color_list[i, 2], class_color_list[i, 1], class_color_list[i, 0]), axis=-1)
        label[i] = feature[:, :] / 1.0

    return label

# Make segmentation label a classifciation label
def label_to_classes(label):
    classifcation_label = np.zeros(class_count)
    for i in range(0, class_count):
        if np.sum(label[i]) > 0:
            classifcation_label[i] = 1

    return classifcation_label

def label_smoothing(inputs):
    # Label smoothing
    # https://arxiv.org/pdf/1906.02629.pdf
    inputs[inputs == 0] = 0.1
    inputs[inputs == 1] = 1 - 0.1

    return inputs

# Dataset to load the voc subset of coco
class VOCOClassification(Dataset):
    def __init__(self, source='train'):
        dataset_root = '../datasets/voco/'
        self.image_root = '../datasets/voco/images/'
        self.label_root = '../datasets/voco/labels/'
        
        # Gather a list of images and labels
        images = read_file(dataset_root + source + '.txt')
        labels = images

        # Save some config info for later us
        self.source = source
        self.images = images
        self.labels = labels
        self.total = len(self.labels)

    def __len__(self):
        return self.total

    def __getitem__(self, sample):
        # Read images and perform augmentation
        image_name = self.labels[sample]
        image = cv2.imread(self.image_root + image_name + '.jpg')
        label = cv2.imread(self.label_root + image_name + '.png')

        # Perform augmentation and normalization.
        # This is crappy, I normally use the albumentations lib, but wanted to keep dependancies low
        image = image / 255.0
        image = cv2.resize(image, (256, 256))
        label = cv2.resize(label, (256, 256))

        # Reorder image to (channel, width, height)
        image = np.moveaxis(image, 2, 0)

        # Transform RGB label into n_channel segmentation label
        label = image_to_label(label)

        # Transform segmentation label into classification label (removing background)
        label = np.delete(label_to_classes(label), 0)

        # Smooth label to ensure gradient
        label = label_smoothing(label)

        return (image, label)
