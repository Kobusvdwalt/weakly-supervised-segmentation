
import enum
import sys, os
sys.path.insert(0, os.path.abspath('../'))

from data import voc2012, cityscapes
from models.vgg_gap import Vgg16GAP
from models.unet import UNet

class Datasets(enum.Enum):
    voc2012 = 1
    cityscapes = 2
    coco = 3

class Models(enum.Enum):
    Vgg16GAP = 1
    Unet = 2

def get_model(dataset, model):
    class_count = None
    model_constructor = None
    name = None

    # Class count
    if dataset == Datasets.voc2012:
        class_count = voc2012.get_class_count()
    if dataset == Datasets.cityscapes:
        class_count = cityscapes.get_class_count()

    # Model constructor
    if model == Models.Vgg16GAP:
        model_constructor = Vgg16GAP
    if model == Models.Unet:
        model_constructor = UNet
        class_count += 1 # TODO: this can be done a little more elagantly

    name = model.name + '_' + dataset.name

    return model_constructor(name, class_count)
