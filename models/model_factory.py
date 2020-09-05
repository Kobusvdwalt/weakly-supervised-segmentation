import enum
from models.vgg_gap import Vgg16GAP

class Datasets(enum.Enum):
    voc2012 = 1
    cityscapes = 2
class Models(enum.Enum):
    Vgg16GAP = 1
    Vgg16GAPup = 2

def get_model(dataset, model):
    if dataset == Datasets.voc2012:
        if model == Models.Vgg16GAP:
            return Vgg16GAP('Vgg16GAP', 20)

        if model == Models.Vgg16GAPup:
            return Vgg16GAP('Vgg16GAPup', 20)
    
    if dataset == Datasets.cityscapes:
        if model == Models.Vgg16GAP:
            return Vgg16GAP('Vgg16GAP', 30)