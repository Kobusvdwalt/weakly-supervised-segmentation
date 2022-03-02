
import numpy as np
import cv2, os, shutil, random

import sys
sys.path.append('../')

from pycocotools.coco import COCO
from data.voc2012 import label_to_image
from os import path

class_count_target = 5000

class VOC_COCO_Bridge():
    def __init__(self, coco):
        self.coco = coco
        self.voc_names_to_coco_names = {
            'aeroplane': 'airplane',
            'bicycle': 'bicycle',
            'bird': 'bird',
            'boat': 'boat',
            'bottle': 'bottle',
            'bus': 'bus',
            'car': 'car',
            'cat': 'cat',
            'chair': 'chair',
            'cow': 'cow',
            'diningtable': 'dining table',
            'dog': 'dog',
            'horse': 'horse',
            'motorbike': 'motorcycle',
            'person': 'person',
            'pottedplant': 'potted plant',
            'sheep': 'sheep',
            'sofa': 'couch',
            'train': 'train',
            'tvmonitor': 'tv'
        }
        self.coco_names_to_voc_names = {}
        for key in self.voc_names_to_coco_names.keys():
            self.coco_names_to_voc_names[self.voc_names_to_coco_names[key]] = key

        self.coco_names = list(self.voc_names_to_coco_names.values())
        self.voc_names = list(self.voc_names_to_coco_names.keys())

        self.coco_names_to_ids = {}
        for cat_name in self.coco_names:
            self.coco_names_to_ids[cat_name] = coco.getCatIds(catNms = [cat_name])[0]
        self.coco_ids = list(self.coco_names_to_ids.values())

    def get_images(self):
        img_ids = {}
        cats_ids = {}
        for cat_id in self.coco_ids:
            cat_img_ids = self.coco.getImgIds(catIds = [cat_id])
            cats_ids[cat_id] = []
            for cat_img_index, img_id in enumerate(cat_img_ids):
                if cat_img_index >= class_count_target:
                    break
                img_ids[img_id] = 1
                cats_ids[cat_id].append(img_id)

        img_ids = list(img_ids.keys())

        return img_ids, cats_ids



def generate():
    random.seed(2219677)
    # Set up bridge between coco and voc
    annFile = 'source/coco/annotations/instances_train2017.json'
    coco = COCO(annFile)
    bridge = VOC_COCO_Bridge(coco)

    # Get image id list
    img_ids, cats_ids = bridge.get_images()

    # Prepare destination folder
    dataset_name = 'voco'
    if path.isdir(f'generated/{dataset_name}'):
        shutil.rmtree(f'generated/{dataset_name}')

    os.makedirs(f'generated/{dataset_name}')
    os.makedirs(f'generated/{dataset_name}/images')
    os.makedirs(f'generated/{dataset_name}/labels')

    # Write images, and id lists
    train_file = open(f'generated/{dataset_name}/train.txt', 'w')
    val_file = open(f'generated/{dataset_name}/val.txt', 'w')

    # Write balanced sampling id list
    cats_ids_train = {}
    cats_ids_val = {}
    for cat_id in cats_ids.keys():
        cats_ids_train[cat_id] = []
        cats_ids_val[cat_id] = []
        for img_inx, img_id in enumerate(cats_ids[cat_id]):
            if img_inx % 10 == 0:
                cats_ids_val[cat_id].append(img_id)
            else:
                cats_ids_train[cat_id].append(img_id)
        
        class_count_current = len(cats_ids_train[cat_id])
        for i in range(class_count_current, class_count_target):
            cats_ids_train[cat_id].append(cats_ids_train[cat_id][random.randint(0, class_count_current-1)])

        print('train', cat_id, len(cats_ids_train[cat_id]), class_count_current)
        for img_id in cats_ids_train[cat_id]:
            train_file.write(f'{img_id}\n')

        for img_id in cats_ids_val[cat_id]:
            val_file.write(f'{img_id}\n')

    # Write images
    for img_index, img_id in enumerate(img_ids):
        img_filename = coco.loadImgs(ids = [img_id])[0]['file_name']
        img = cv2.imread(f'source/coco/train2017/{img_filename}')
        
        ans_ids = coco.getAnnIds(imgIds=[img_id])
        ans = coco.loadAnns(ans_ids)

        label = np.zeros((21, img.shape[0], img.shape[1]))
        for an in ans:
            an_cat_id = an['category_id']
            if an_cat_id in bridge.coco_ids:
                mask = coco.annToMask(an)
                voc_index = bridge.coco_ids.index(an_cat_id)
                label[voc_index+1] += mask
        label_rgb = label_to_image(label)

        cv2.imwrite(f'generated/{dataset_name}/images/{img_id}.jpg', img)
        cv2.imwrite(f'generated/{dataset_name}/labels/{img_id}.png', label_rgb * 255.0)
        print(img_index)

    train_file.close()
    val_file.close()

generate()