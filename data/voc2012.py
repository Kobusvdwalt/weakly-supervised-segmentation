
import numpy as np

class_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_count = len(class_list)

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
# Output: 3 dim tensor (x, y, class_count)
def ImageToLabel(image):
    label = np.zeros((class_count, image.shape[0], image.shape[1]))
    for i in range(0, class_count):
        feature = np.all(image == (class_color_list[i, 2], class_color_list[i, 1], class_color_list[i, 0]), axis=-1)
        label[i] = feature[:, :] / 1.0

    return label

# Input : RGB image (x, y, 3)
# Output: 1 dim vector of (class_count)
def ImageToClasses(image):
    classifcation_label = np.zeros(class_count)
    for i in range(0, class_count):
        feature = np.all(image == (class_color_list[i, 2], class_color_list[i, 1], class_color_list[i, 0]), axis=-1)

        if np.sum(feature) > 0:
            classifcation_label[i] = 1

    return classifcation_label


# Input : 3 dim tensor (x, y, class_count)
# Output: 1 dim vector of (class_count)
def LabelToClasses(label):
    classifcation_label = np.zeros(class_count)
    for i in range(0, class_count):
        if np.sum(label[i]) > 0:
            classifcation_label[i] = 1

    return classifcation_label

# Input : One-hot encoded vector
# Output: List of corresponding class descriptions
def ClassesToWords(classes):
    words = []
    for i in range(0, class_count):
        if (classes[i] == 1):
            words.append(class_list[i])

    return words

# Input : Vector of class scores
# Output: One-hot encoded vector
def ThresholdClasses(classes, threshold=0.5):
    classes[classes > threshold] = 1
    classes[classes <= threshold] = 0
    return classes

# Input : Vector without background class
# Output: Vector with background class
def AddBackgroundClass(classes):
    background = np.insert(classes, 0, 1, axis=0)
    return background

