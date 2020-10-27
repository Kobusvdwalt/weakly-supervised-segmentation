
import sys, os

sys.path.insert(0, os.path.abspath('../'))

from visualize.visualize import visualize
from torch.utils.data.dataloader import DataLoader
from data.voc2012_loaders import PascalVOCSegmentation
from models.vgg11_gap_feat import Vgg11GAP
from models.vgg13_gap_feat import Vgg13GAP
from models.vgg16_gap_feat import Vgg16GAP

from data import voc2012

model = Vgg11GAP('voc_classification', voc2012.get_class_count() -1)
model.load()
visualize(
    model=model,
    dataloaders = {
        'val': DataLoader(PascalVOCSegmentation('val'), batch_size=1, shuffle=False, num_workers=0),
    },
    palette = voc2012.color_map(256)
)

model = Vgg13GAP('voc_classification', voc2012.get_class_count() -1)
model.load()
visualize(
    model=model,
    dataloaders = {
        'val': DataLoader(PascalVOCSegmentation('val'), batch_size=1, shuffle=False, num_workers=0),
    },
    palette = voc2012.color_map(256)
)

model = Vgg16GAP('voc_classification', voc2012.get_class_count() -1)
model.load()
visualize(
    model=model,
    dataloaders = {
        'val': DataLoader(PascalVOCSegmentation('val'), batch_size=1, shuffle=False, num_workers=0),
    },
    palette = voc2012.color_map(256)
)