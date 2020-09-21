
import sys, os
sys.path.insert(0, os.path.abspath('../'))

from visualize.visualize import visualize
from models.model_factory import Datasets, Models
from data.loader_factory import LoaderSplit, LoaderType

visualize(
    Models.Vgg16GAP,
    Datasets.voc2012,
    LoaderSplit.val
)