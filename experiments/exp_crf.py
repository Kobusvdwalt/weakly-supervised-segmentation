from data.voc2012 import image_to_label
from data.voc2012 import class_word_to_index

def start():
    import cv2
    import torch
    import numpy as np

    from tools.crf import CRF

    crf = CRF()

    image = cv2.imread('./datasets/voco/529.jpg', 1)
    mask = cv2.imread('./datasets/voco/529.png')
    mask = image_to_label(mask)

    result = crf.process(image, mask)

    cv2.imshow('q', result[class_word_to_index('person')])
    cv2.imshow('q2', result[class_word_to_index('motorbike')])
    cv2.imshow('image', image)
    cv2.waitKey(0)