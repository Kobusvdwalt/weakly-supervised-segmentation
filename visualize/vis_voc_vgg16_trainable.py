
import sys, os

sys.path.insert(0, os.path.abspath('../'))

from visualize.visualize import visualize
from torch.utils.data.dataloader import DataLoader
from data.voc2012_loaders import PascalVOCSegmentation
from models.vgg16_gap_feat_unfreeze import Vgg16GAPUnfreeze
from data import voc2012

model = Vgg16GAPUnfreeze('voc_classification', voc2012.get_class_count() -1, 0)
model.load()
visualize(
    model=model,
    dataloaders = {
        'val': DataLoader(PascalVOCSegmentation('val'), batch_size=1, shuffle=False, num_workers=0),
    },
    palette = voc2012.color_map(256)
)

model = Vgg16GAPUnfreeze('voc_classification', voc2012.get_class_count() -1, 1)
model.load()
visualize(
    model=model,
    dataloaders = {
        'val': DataLoader(PascalVOCSegmentation('val'), batch_size=1, shuffle=False, num_workers=0),
    },
    palette = voc2012.color_map(256)
)

model = Vgg16GAPUnfreeze('voc_classification', voc2012.get_class_count() -1, 2)
model.load()
visualize(
    model=model,
    dataloaders = {
        'val': DataLoader(PascalVOCSegmentation('val'), batch_size=1, shuffle=False, num_workers=0),
    },
    palette = voc2012.color_map(256)
)

model = Vgg16GAPUnfreeze('voc_classification', voc2012.get_class_count() -1, 3)
model.load()
visualize(
    model=model,
    dataloaders = {
        'val': DataLoader(PascalVOCSegmentation('val'), batch_size=1, shuffle=False, num_workers=0),
    },
    palette = voc2012.color_map(256)
)