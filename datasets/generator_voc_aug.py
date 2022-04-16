import os
import os.path
import shutil
from os import path

import cv2
import numpy as np

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

def label_to_image(label):
    image = np.zeros((label.shape[1], label.shape[2], 3))

    mask_a = np.argmax(label, 0)
    image[:, :, 0] = class_color_list[mask_a, 2] / 255
    image[:, :, 1] = class_color_list[mask_a, 1] / 255
    image[:, :, 2] = class_color_list[mask_a, 0] / 255

    return image

def generate():
    dataset_name = 'voc_aug'
    if path.isdir(f'generated/{dataset_name}'):
        shutil.rmtree(f'generated/{dataset_name}')

    os.makedirs(f'generated/{dataset_name}')
    os.makedirs(f'generated/{dataset_name}/images')
    os.makedirs(f'generated/{dataset_name}/labels')

    train_aug_lines = open("source/VOC2012/ImageSets/train_aug.txt", "r").readlines()
    image_names = []

    for tl in train_aug_lines:
        image_path = tl.split(' ')[0]
        image_name = image_path.split('/')[2].split('.')[0]
        image_names.append(image_name)

        shutil.copyfile(f'source/VOC2012/JPEGImages/{image_name}.jpg', f'generated/{dataset_name}/images/{image_name}.jpg')

        # For each label, transform into correct format
        label_image = cv2.imread(f'source/VOC2012/SegmentationClassAug/{image_name}.png', cv2.IMREAD_GRAYSCALE)
        border_indices = label_image == 255
        label_np = np.zeros((21, label_image.shape[0], label_image.shape[1]))
        for i in range(0, 21):
            label_np[i][label_image == i] = 1

        label_image_out = label_to_image(label_np)
        label_image_out[border_indices, 0] = 192/255
        label_image_out[border_indices, 1] = 224/255
        label_image_out[border_indices, 2] = 224/255

        cv2.imwrite(f'generated/{dataset_name}/labels/{image_name}.png', label_image_out * 255)

        # cv2.imshow('label_image', label_image)
        # cv2.imshow('label_image_out', label_image_out)
        # cv2.waitKey(1)

    image_names_with_newlines = '\n'.join(image_names) + '\n'

    train_file = open(f'generated/{dataset_name}/train.txt', 'w')
    train_file.write(image_names_with_newlines)
    train_file.close()

    val_file = open(f'generated/{dataset_name}/val.txt', 'w')
    val_file.write(image_names_with_newlines)
    val_file.close()

generate()