import sys, os
sys.path.insert(0, os.path.abspath('../'))

from data.cityscapes_loaders import CityscapesClassification, CityscapesSegmentation
from data.voc2012_loaders import *
from models.model_factory import Datasets


import enum
class LoaderType(enum.Enum):
    classification = 1
    segmentation = 2

class LoaderSplit(enum.Enum):
    train = 1
    val = 2
    test = 3

def get_loader(dataset, type, split):
    if (dataset == Datasets.voc2012):
        if (type == LoaderType.classification):
            if (split == LoaderSplit.train):
                return PascalVOCClassificationMulticlass('train')
            if (split == LoaderSplit.val):
                return PascalVOCClassificationMulticlass('val')
            if (split == LoaderSplit.test):
                return PascalVOCClassificationMulticlass('test')
        
        if (type == LoaderType.segmentation):
            if (split == LoaderSplit.train):
                return PascalVOCSegmentation('train')
            if (split == LoaderSplit.val):
                return PascalVOCSegmentation('val')
            if (split == LoaderSplit.test):
                return PascalVOCSegmentation('test')

    if (dataset == Datasets.cityscapes):
        if (type == LoaderType.classification):
            if (split == LoaderSplit.train):
                return CityscapesClassification('train')
            if (split == LoaderSplit.val):
                return CityscapesClassification('val')
            if (split == LoaderSplit.test):
                return CityscapesClassification('test')
        
        if (type == LoaderType.segmentation):
            if (split == LoaderSplit.train):
                return CityscapesSegmentation('train')
            if (split == LoaderSplit.val):
                return CityscapesSegmentation('val')
            if (split == LoaderSplit.test):
                return CityscapesSegmentation('test')