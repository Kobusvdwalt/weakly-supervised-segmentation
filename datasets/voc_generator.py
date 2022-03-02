import os
import os.path
import shutil
from os import path

def generate():
    dataset_name = 'voc'
    if path.isdir(f'generated/{dataset_name}'):
        shutil.rmtree(f'generated/{dataset_name}')

    os.makedirs(f'generated/{dataset_name}')
    os.makedirs(f'generated/{dataset_name}/images')
    os.makedirs(f'generated/{dataset_name}/labels')

    train_lines = open("source/VOC2012/ImageSets/Segmentation/train.txt", "r").readlines()
    val_lines = open("source/VOC2012/ImageSets/Segmentation/val.txt", "r").readlines()
    train_val_lines = open("source/VOC2012/ImageSets/Segmentation/trainval.txt", "r").readlines()

    train_file = open(f'generated/{dataset_name}/train.txt', 'w')
    train_file.writelines(train_lines)
    train_file.close()

    val_file = open(f'generated/{dataset_name}/val.txt', 'w')
    val_file.writelines(val_lines)
    val_file.close()

    train_val_file = open(f'generated/{dataset_name}/trainval.txt', 'w')
    train_val_file.writelines(train_val_lines)
    train_val_file.close()

    for tl in train_lines:
        file = tl.replace('\n', '')
        shutil.copyfile(f'source/VOC2012/JPEGImages/{file}.jpg', f'generated/{dataset_name}/images/{file}.jpg')
        shutil.copyfile(f'source/VOC2012/SegmentationClass/{file}.png', f'generated/{dataset_name}/labels/{file}.png')

    for tl in val_lines:
        file = tl.replace('\n', '')
        shutil.copyfile(f'source/VOC2012/JPEGImages/{file}.jpg', f'generated/{dataset_name}/images/{file}.jpg')
        shutil.copyfile(f'source/VOC2012/SegmentationClass/{file}.png', f'generated/{dataset_name}/labels/{file}.png')

generate()